#include "keren/Interpreter.h"
#include "keren/LinalgHandler.h"
#include "keren/StableHLOHandler.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Tensor.h"
#include "stablehlo/reference/Value.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <sstream>

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                  llvm::cl::desc("<input.mlir>"),
                                                  llvm::cl::Required);

static llvm::cl::opt<bool>
    lowerToLinalg("lower-to-linalg",
                  llvm::cl::desc("Lower StableHLO to Linalg before interpreting"),
                  llvm::cl::init(false));

static llvm::cl::opt<std::string>
    entryFunction("entry",
                  llvm::cl::desc("Entry function name (default: main)"),
                  llvm::cl::init("main"));

static llvm::cl::opt<bool>
    jsonOutput("json",
               llvm::cl::desc("Output results as JSON for visualizer consumption"),
               llvm::cl::init(false));

static llvm::cl::opt<std::string>
    traceDir("trace",
             llvm::cl::desc("Directory for per-op probe instrumentation output"),
             llvm::cl::init(""));

static llvm::cl::list<std::string>
    inputValues("input",
                llvm::cl::desc("Input tensor as JSON array (one per argument, in order)"),
                llvm::cl::ZeroOrMore);

static llvm::cl::opt<std::string>
    inputFile("input-file",
              llvm::cl::desc("JSON file with input values (array of tensors)"),
              llvm::cl::init(""));

// --- JSON input parsing ---

/// Flatten a JSON array into a list of double values (works for ints too).
static bool flattenJsonArray(const llvm::json::Value &val,
                             std::vector<double> &out) {
  if (auto *arr = val.getAsArray()) {
    for (const auto &elem : *arr) {
      if (!flattenJsonArray(elem, out))
        return false;
    }
    return true;
  }
  if (auto d = val.getAsNumber()) {
    out.push_back(*d);
    return true;
  }
  if (auto i = val.getAsInteger()) {
    out.push_back(static_cast<double>(*i));
    return true;
  }
  return false;
}

/// Infer shape from a nested JSON array.
static bool inferShape(const llvm::json::Value &val,
                       llvm::SmallVectorImpl<int64_t> &shape) {
  if (auto *arr = val.getAsArray()) {
    shape.push_back(arr->size());
    if (!arr->empty())
      return inferShape((*arr)[0], shape);
    return true;
  }
  // Scalar element — stop recursion.
  return true;
}

/// Parse a JSON string into an InterpreterValue given the expected MLIR type.
static llvm::Expected<mlir::stablehlo::InterpreterValue>
parseInputTensor(llvm::StringRef jsonStr, mlir::ShapedType expectedType,
                 mlir::MLIRContext *ctx) {
  auto parsed = llvm::json::parse(jsonStr);
  if (!parsed)
    return parsed.takeError();

  std::vector<double> flat;
  if (!flattenJsonArray(*parsed, flat))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to parse JSON array of numbers");

  auto elemType = expectedType.getElementType();
  int64_t numElements = expectedType.getNumElements();

  if (static_cast<int64_t>(flat.size()) != numElements)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Input has %zu elements but type %s expects %lld",
        flat.size(), "", numElements);

  mlir::DenseElementsAttr attr;
  if (elemType.isInteger(32)) {
    llvm::SmallVector<int32_t> vals;
    vals.reserve(flat.size());
    for (double v : flat) vals.push_back(static_cast<int32_t>(v));
    attr = mlir::DenseElementsAttr::get(expectedType, llvm::ArrayRef(vals));
  } else if (elemType.isInteger(64)) {
    llvm::SmallVector<int64_t> vals;
    vals.reserve(flat.size());
    for (double v : flat) vals.push_back(static_cast<int64_t>(v));
    attr = mlir::DenseElementsAttr::get(expectedType, llvm::ArrayRef(vals));
  } else if (elemType.isInteger(16)) {
    llvm::SmallVector<int16_t> vals;
    vals.reserve(flat.size());
    for (double v : flat) vals.push_back(static_cast<int16_t>(v));
    attr = mlir::DenseElementsAttr::get(expectedType, llvm::ArrayRef(vals));
  } else if (elemType.isInteger(8)) {
    llvm::SmallVector<int8_t> vals;
    vals.reserve(flat.size());
    for (double v : flat) vals.push_back(static_cast<int8_t>(v));
    attr = mlir::DenseElementsAttr::get(expectedType, llvm::ArrayRef(vals));
  } else if (elemType.isF32()) {
    llvm::SmallVector<float> vals;
    vals.reserve(flat.size());
    for (double v : flat) vals.push_back(static_cast<float>(v));
    attr = mlir::DenseElementsAttr::get(expectedType, llvm::ArrayRef(vals));
  } else if (elemType.isF64()) {
    llvm::SmallVector<double> vals(flat.begin(), flat.end());
    attr = mlir::DenseElementsAttr::get(expectedType, llvm::ArrayRef(vals));
  } else {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Unsupported element type for input");
  }

  auto tensor = mlir::stablehlo::makeTensor(attr);
  return mlir::stablehlo::InterpreterValue(tensor);
}

/// Parse all --input values using the entry function's argument types.
static llvm::Expected<llvm::SmallVector<mlir::stablehlo::InterpreterValue>>
parseAllInputs(mlir::ModuleOp module, llvm::StringRef entryName,
               const std::vector<std::string> &jsonInputs) {
  auto func = module.lookupSymbol<mlir::func::FuncOp>(entryName);
  if (!func)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Entry function '%s' not found",
                                   entryName.str().c_str());

  auto funcType = func.getFunctionType();
  if (jsonInputs.size() != static_cast<size_t>(funcType.getNumInputs()))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Function '%s' expects %d inputs but %zu provided",
        entryName.str().c_str(), funcType.getNumInputs(), jsonInputs.size());

  llvm::SmallVector<mlir::stablehlo::InterpreterValue> inputs;
  for (unsigned i = 0; i < jsonInputs.size(); ++i) {
    auto shapedType = mlir::dyn_cast<mlir::ShapedType>(funcType.getInput(i));
    if (!shapedType)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Argument %d is not a shaped type", i);

    auto valOrErr = parseInputTensor(jsonInputs[i], shapedType,
                                     module->getContext());
    if (!valOrErr)
      return valOrErr.takeError();

    inputs.push_back(std::move(*valOrErr));
  }
  return inputs;
}

/// Escape a string for JSON (handles newlines, quotes, backslashes).
static std::string jsonEscape(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 16);
  for (char c : s) {
    switch (c) {
    case '"':  out += "\\\""; break;
    case '\\': out += "\\\\"; break;
    case '\n': out += "\\n";  break;
    case '\r': out += "\\r";  break;
    case '\t': out += "\\t";  break;
    default:   out += c;      break;
    }
  }
  return out;
}

static void printJSON(const std::vector<keren::TensorValue> &results) {
  llvm::outs() << "{\n";
  for (size_t i = 0; i < results.size(); ++i) {
    // Capture type to string
    std::string typeStr;
    llvm::raw_string_ostream typeOS(typeStr);
    results[i].getType().print(typeOS);

    // Capture value to string
    std::string valStr;
    llvm::raw_string_ostream valOS(valStr);
    results[i].print(valOS);

    llvm::outs() << "  \"result_" << i << "\": {"
                 << "\"type\": \"" << jsonEscape(typeStr) << "\", "
                 << "\"value\": \"" << jsonEscape(valStr) << "\"}";
    if (i + 1 < results.size()) llvm::outs() << ",";
    llvm::outs() << "\n";
  }
  llvm::outs() << "}\n";
}

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "Keren ML Graph Compiler Simulator\n");

  keren::InterpreterOptions opts;
  opts.entryFunction = entryFunction;
  opts.lowerToLinalg = lowerToLinalg;
  opts.probeDir = traceDir;

  // Collect input JSON strings from --input flags and/or --input-file.
  std::vector<std::string> jsonInputs(inputValues.begin(), inputValues.end());

  if (!inputFile.empty()) {
    std::ifstream ifs(inputFile.getValue());
    if (!ifs) {
      llvm::errs() << "error: cannot open input file: " << inputFile << "\n";
      return 1;
    }
    std::stringstream ss;
    ss << ifs.rdbuf();
    auto parsed = llvm::json::parse(ss.str());
    if (!parsed) {
      llvm::errs() << "error: invalid JSON in input file: "
                    << llvm::toString(parsed.takeError()) << "\n";
      return 1;
    }
    if (auto *arr = parsed->getAsArray()) {
      for (const auto &elem : *arr) {
        std::string s;
        llvm::raw_string_ostream os(s);
        os << elem;
        jsonInputs.push_back(s);
      }
    } else {
      llvm::errs() << "error: input file must contain a JSON array\n";
      return 1;
    }
  }

  keren::Interpreter interp(opts);
  interp.addHandler(std::make_unique<keren::StableHLOHandler>());
  interp.addHandler(std::make_unique<keren::LinalgHandler>());

  // If inputs are provided, parse the module using the interpreter's context
  // to get argument types, then construct InterpreterValues.
  llvm::SmallVector<mlir::stablehlo::InterpreterValue> inputs;
  if (!jsonInputs.empty()) {
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(
        inputFilename, &interp.getContext());
    if (!module) {
      llvm::errs() << "error: failed to parse " << inputFilename << "\n";
      return 1;
    }

    auto inputsOrErr = parseAllInputs(*module, entryFunction, jsonInputs);
    if (!inputsOrErr) {
      llvm::errs() << "error: " << llvm::toString(inputsOrErr.takeError())
                    << "\n";
      return 1;
    }
    inputs = std::move(*inputsOrErr);
  }

  auto results = interp.runFile(inputFilename, inputs);

  if (jsonOutput) {
    printJSON(results);
  } else {
    for (size_t i = 0; i < results.size(); ++i) {
      llvm::outs() << "Result " << i << ": ";
      results[i].print(llvm::outs());
      llvm::outs() << "\n";
    }
  }

  return results.empty() && !lowerToLinalg ? 1 : 0;
}
