# ruff: noqa: T201, ANN201
import argparse
import time
import tracemalloc
from difflib import SequenceMatcher as DifflibSequenceMatcher

from simple_rcs._myersdiff_dmp import MyersSequenceMatcher as MyersDiffCMatcherC
from simple_rcs._myersdiff_ses import MyersSequenceMatcher as MyersDiffAMatcherA

from simple_rcs.myersdiff_ses import MyersSequenceMatcher
from simple_rcs.pydifflib import StreamTextSequenceMatcher

# FILE_A_PATH = "tmp_mem_test_a.txt"
# FILE_B_PATH = "tmp_mem_test_b.txt"
FILE_A_PATH = "aa.py"
FILE_B_PATH = "bb.py"


def generate_test_files(size_kb: int, num_diff_lines: int):
    """Generates two large text files for comparison."""
    print(f"Generating test files of ~{size_kb}KiB...")
    line_content = "This is a test line with some padding to make it longer. Lorem ipsum dolor sit amet.\n"
    num_lines = (size_kb * 1024) // len(line_content)

    diff_line_indices = set(range(num_lines // 2, num_lines // 2 + num_diff_lines))

    # Generate file A
    with open(FILE_A_PATH, "w") as f:
        for i in range(num_lines):
            f.write(f"{i:08d}: {line_content}")

    # Generate file B
    with open(FILE_B_PATH, "w") as f:
        for i in range(num_lines):
            if i in diff_line_indices:
                f.write(f"{i:08d}: This line has been MODIFIED to show differences.\n")
            else:
                f.write(f"{i:08d}: {line_content}")
    print("Test files generated.")


def measure_difflib():
    """Measures memory usage of the standard difflib.SequenceMatcher."""
    print("\n--- Measuring memory for difflib.SequenceMatcher ---")

    tracemalloc.start()
    start_time = time.perf_counter()

    # Difflib requires loading entire files into memory
    with open(FILE_A_PATH) as f_a, open(FILE_B_PATH) as f_b:
        lines_a = f_a.readlines()
        lines_b = f_b.readlines()

    current, peak_after_read = tracemalloc.get_traced_memory()
    print(f"Memory after reading files into lists: {peak_after_read / 1024:.2f} KiB")

    matcher = DifflibSequenceMatcher(None, lines_a, lines_b, autojunk=False)
    opcodes = matcher.get_opcodes() # This computes the diff

    # opcodes are already computed, this is just to show they are there
    _ = len(opcodes)

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Peak memory usage: {peak / 1024:.2f} KiB")
    print(f"Time taken: {end_time - start_time:.4f} seconds")


def measure_stream_text_sequence_matcher():
    """Measures memory usage of StreamTextSequenceMatcher."""
    print("\n--- Measuring memory for StreamTextSequenceMatcher ---")

    tracemalloc.start()
    start_time = time.perf_counter()

    # StreamTextSequenceMatcher works with file streams, not loading all content
    with open(FILE_A_PATH, "rb") as f_a, open(FILE_B_PATH, "rb") as f_b:
        matcher = StreamTextSequenceMatcher(a_stream=f_a, b_stream=f_b, chunk_size=None)
        # We must consume the generator to perform the comparison
        opcodes = list(matcher.get_opcodes())
        _ = len(opcodes)

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Peak memory usage: {peak / 1024:.2f} KiB")
    print(f"Time taken: {end_time - start_time:.4f} seconds")


def measure_myersdiff_matcher():
    """Measures memory usage of MyersSequenceMatcher (from myersdiff_ses.py)."""
    print("\n--- Measuring memory for MyersSequenceMatcher (myersdiff_ses.py) ---")

    tracemalloc.start()
    start_time = time.perf_counter()

    with open(FILE_A_PATH) as f_a, open(FILE_B_PATH) as f_b:
    #with open(FILE_A_PATH, "rb") as f_a, open(FILE_B_PATH, "rb") as f_b:
        lines_a = [line.rstrip('\r\n') for line in f_a.readlines()]
        lines_b = [line.rstrip('\r\n') for line in f_b.readlines()]

        matcher = MyersSequenceMatcher(a=lines_a, b=lines_b)
        #matcher = MyersStreamSequenceMatcher(a_stream=f_a, b_stream=f_b, chunk_size=None)
        opcodes = list(matcher.get_opcodes())
        _ = len(opcodes)

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Peak memory usage: {peak / 1024:.2f} KiB")
    print(f"Time taken: {end_time - start_time:.4f} seconds")


def measure_myersdiffa_matcher():
    """Measures memory usage of MyersSequenceMatcherA (from myersdiff_.py)."""
    print("\n--- Measuring memory for Cython MyersSequenceMatcherA (_myersdiff_ses.pyx) ---")

    tracemalloc.start()
    start_time = time.perf_counter()

    with open(FILE_A_PATH) as f_a, open(FILE_B_PATH) as f_b:
        lines_a = [line.rstrip('\r\n') for line in f_a.readlines()]
        lines_b = [line.rstrip('\r\n') for line in f_b.readlines()]

    matcher = MyersDiffAMatcherA(a=lines_a, b=lines_b)
    opcodes = list(matcher.get_opcodes())
    _ = len(opcodes)

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Peak memory usage: {peak / 1024:.2f} KiB")
    print(f"Time taken: {end_time - start_time:.4f} seconds")


def measure_myersdiffc_matcher():
    """Measures memory usage of MyersSequenceMatcher (from myersdiffc.py)."""
    print("\n--- Measuring memory for Cython MyersSequenceMatcher (_myersdiff_dmp.pyx) ---")

    tracemalloc.start()
    start_time = time.perf_counter()

    with open(FILE_A_PATH) as f_a, open(FILE_B_PATH) as f_b:
        lines_a = [line.rstrip('\r\n') for line in f_a.readlines()]
        lines_b = [line.rstrip('\r\n') for line in f_b.readlines()]

        matcher = MyersDiffCMatcherC(a=lines_a, b=lines_b)
        opcodes = list(matcher.get_opcodes())
        _ = len(opcodes)

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Peak memory usage: {peak / 1024:.2f} KiB")
    print(f"Time taken: {end_time - start_time:.4f} seconds")


def cleanup_files():
    """Removes the generated test files."""
    print("\nCleaning up temporary files...")
    #if os.path.exists(FILE_A_PATH):
    #    os.remove(FILE_A_PATH)
    #if os.path.exists(FILE_B_PATH):
    #    os.remove(FILE_B_PATH)
    print("Cleanup complete.")


def main():
    parser = argparse.ArgumentParser(description="Compare memory usage of difflib and StreamTextSequenceMatcher.")
    parser.add_argument("--size", type=int, default=60, help="Size of the test files to generate in KiB.")
    parser.parse_args()

    try:
        #generate_test_files(args.size, 5)
        measure_difflib()
        measure_stream_text_sequence_matcher()
        measure_myersdiff_matcher()
        measure_myersdiffa_matcher()
        measure_myersdiffc_matcher()
    finally:
        #cleanup_files()
        pass


if __name__ == "__main__":
    main()
