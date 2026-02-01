#include "keren/Interpreter.h"
#include "keren/LinalgHandler.h"
#include "keren/StableHLOHandler.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

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

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "Keren ML Graph Compiler Simulator\n");

  keren::InterpreterOptions opts;
  opts.entryFunction = entryFunction;
  opts.lowerToLinalg = lowerToLinalg;

  keren::Interpreter interp(opts);
  interp.addHandler(std::make_unique<keren::StableHLOHandler>());
  interp.addHandler(std::make_unique<keren::LinalgHandler>());

  auto results = interp.runFile(inputFilename);

  for (size_t i = 0; i < results.size(); ++i) {
    llvm::outs() << "Result " << i << ": ";
    results[i].print(llvm::outs());
    llvm::outs() << "\n";
  }

  return results.empty() && !lowerToLinalg ? 1 : 0;
}
