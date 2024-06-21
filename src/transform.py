import math

class HilbertMapper:
    def __init__(self, size):
        self.size = size
        self.curve = self._hilbert_curve(size)

    def _rotate(self, n, x, y, rx, ry):
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        return x, y

    def _point_to_hilbert(self, x, y):
        d = 0
        s = self.size // 2
        while s > 0:
            rx = (x & s) > 0
            ry = (y & s) > 0
            d += s * s * ((3 * rx) ^ ry)
            x, y = self._rotate(s, x, y, rx, ry)
            s //= 2
        return d

    def _hilbert_curve(self, n):
        points = [(x, y) for y in range(n) for x in range(n)]
        return sorted(points, key=lambda p: self._point_to_hilbert(p[0], p[1]))

    def flatten(self, matrix):
        if len(matrix) != self.size or len(matrix[0]) != self.size:
            raise ValueError(f"Matrix must be {self.size}x{self.size}")
        return [matrix[y][x] for x, y in self.curve]

    def unflatten(self, flattened):
        if len(flattened) != self.size * self.size:
            raise ValueError(f"Flattened list must have {self.size * self.size} elements")
        matrix = [[None for _ in range(self.size)] for _ in range(self.size)]
        for (x, y), value in zip(self.curve, flattened):
            matrix[y][x] = value
        return matrix

# Test cases
def run_tests():
    # Test case 1: 2x2 matrix
    mapper_2x2 = HilbertMapper(2)
    matrix_2x2 = [[1, 2], [3, 4]]
    flattened_2x2 = mapper_2x2.flatten(matrix_2x2)
    assert flattened_2x2 == [1, 2, 3, 4], f"Test case 1 flatten failed: {flattened_2x2}"
    unflattened_2x2 = mapper_2x2.unflatten(flattened_2x2)
    assert unflattened_2x2 == matrix_2x2, f"Test case 1 unflatten failed: {unflattened_2x2}"
    print("Test case 1 passed")

    # Test case 2: 4x4 matrix
    mapper_4x4 = HilbertMapper(4)
    matrix_4x4 = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]
    flattened_4x4 = mapper_4x4.flatten(matrix_4x4)
    expected_4x4 = [1, 2, 5, 6, 7, 8, 3, 4, 13, 14, 9, 10, 11, 12, 15, 16]
    assert flattened_4x4 == expected_4x4, f"Test case 2 flatten failed: {flattened_4x4}"
    unflattened_4x4 = mapper_4x4.unflatten(flattened_4x4)
    assert unflattened_4x4 == matrix_4x4, f"Test case 2 unflatten failed: {unflattened_4x4}"
    print("Test case 2 passed")

    # Test case 3: 8x8 matrix with 'UNUSED' padding
    mapper_8x8 = HilbertMapper(8)
    matrix_8x8 = [['UNUSED' for _ in range(8)] for _ in range(8)]
    for i in range(4):
        for j in range(4):
            matrix_8x8[i][j] = i * 4 + j + 1
    flattened_8x8 = mapper_8x8.flatten(matrix_8x8)
    unflattened_8x8 = mapper_8x8.unflatten(flattened_8x8)
    assert unflattened_8x8 == matrix_8x8, f"Test case 3 failed"
    print("Test case 3 passed")

    # Test case 4: Error handling
    mapper_4x4 = HilbertMapper(4)
    try:
        mapper_4x4.flatten([[1, 2], [3, 4]])
        assert False, "Test case 4 failed: Expected ValueError"
    except ValueError:
        print("Test case 4 passed")

    try:
        mapper_4x4.unflatten([1, 2, 3, 4])
        assert False, "Test case 5 failed: Expected ValueError"
    except ValueError:
        print("Test case 5 passed")

if __name__ == "__main__":
    run_tests()
        
