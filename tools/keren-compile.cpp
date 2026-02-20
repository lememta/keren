//===- keren-compile.cpp - StableHLO to Linalg compiler CLI ---------------===//
//
// This file implements the command-line interface for the Keren compiler.
//
// Usage:
//   keren-compile [options] <input.mlir>
//
// Options:
//   --lower-ops        Enable op-by-op lowering (default: true)
//   --fuse             Enable elementwise fusion after lowering
//   --verbose          Print verbose output showing each compilation stage
//   --output=<file>    Write output to file (default: stdout)
//
// Example:
//   keren-compile --fuse input.mlir -o output.mlir
//
//===----------------------------------------------------------------------===//

#include "keren/Compiler.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::opt<std::string>
    inputFilename(llvm::cl::Positional, llvm::cl::desc("<input.mlir>"),
                  llvm::cl::Required);

static llvm::cl::opt<bool>
    lowerOps("lower-ops",
             llvm::cl::desc("Enable op-by-op lowering from StableHLO to Linalg"),
             llvm::cl::init(true));

static llvm::cl::opt<bool>
    fuse("fuse",
         llvm::cl::desc("Enable elementwise fusion after lowering"),
         llvm::cl::init(false));

static llvm::cl::opt<bool>
    verbose("verbose",
            llvm::cl::desc("Print verbose output showing each compilation stage"),
            llvm::cl::init(false));

static llvm::cl::opt<std::string>
    outputFile("o", llvm::cl::desc("Output file (default: stdout)"),
               llvm::cl::value_desc("filename"), llvm::cl::init(""));

static llvm::cl::alias
    outputFileAlias("output", llvm::cl::aliasopt(outputFile),
                    llvm::cl::desc("Alias for -o"));

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "Keren Compiler - StableHLO to Linalg with Op-by-Op Lowering and Fusion\n\n"
      "This tool compiles StableHLO MLIR programs to the Linalg dialect.\n"
      "Each StableHLO operation is lowered individually, with optional\n"
      "elementwise operation fusion to reduce memory bandwidth.\n\n"
      "Example:\n"
      "  keren-compile --fuse input.mlir -o output.mlir\n");

  // Build compiler options
  keren::CompilerOptions opts;
  opts.lowerOps = lowerOps;
  opts.fuse = fuse;
  opts.verbose = verbose;
  opts.outputFile = outputFile;

  // Create compiler and run
  keren::Compiler compiler(opts);
  auto module = compiler.compileFile(inputFilename);

  if (!module) {
    return 1;
  }

  // Print output
  compiler.printOutput(*module);

  return 0;
}
