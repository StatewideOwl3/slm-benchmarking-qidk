from codecarbon import EmissionsTracker
import random


def create_matrix(n: int):
    """Create an n x n matrix filled with random floats."""
    return [[random.random() for _ in range(n)] for _ in range(n)]


def matrix_multiply(A, B):
    """Naive matrix multiplication A x B."""
    n = len(A)
    result = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return result


def main():
    size = 100  # increase this if you want heavier computation
    A = create_matrix(size)

    # Set up CodeCarbon tracker
    tracker = EmissionsTracker(project_name="dummy_matrix_square")

    # Start tracking
    tracker.start()

    # Operation to be measured: square the matrix (A x A)
    squared = matrix_multiply(A, A)

    # Stop tracking
    emissions = tracker.stop()

    # Simple outputs so you can see it worked
    print(f"Matrix size: {size}x{size}")
    print(f"Squared[0][0] = {squared[0][0]}")
    print(f"Estimated emissions: {emissions:.8f} kg CO2eq")


if __name__ == "__main__":
    main()


