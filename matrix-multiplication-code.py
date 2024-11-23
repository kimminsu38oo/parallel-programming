import numpy as np
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import time

class MatrixMultiplication:
    def __init__(self, A, B):
        self.A = np.array(A)
        self.B = np.array(B)
        # 결과 행렬 초기화
        self.C = np.zeros((self.A.shape[0], self.B.shape[1]))
        
    def compute_column(self, col_idx):
        """B의 특정 열과 A의 곱을 계산"""
        b_col = self.B[:, col_idx]
        result = np.zeros(self.A.shape[0])
        
        # A와 B의 열벡터의 곱 계산
        for i in range(self.A.shape[0]):
            for k in range(self.A.shape[1]):
                result[i] += self.A[i,k] * b_col[k]
                
        self.C[:, col_idx] = result
    
    def parallel_multiply(self, num_threads=4):
        """병렬 처리로 행렬 곱셈 수행"""
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # B의 각 열에 대해 병렬 처리
            executor.map(self.compute_column, range(self.B.shape[1]))
        return self.C

# 테스트
def test_multiplication():
    # 테스트용 행렬
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    
    B = np.array([[9, 8, 7],
                  [6, 5, 4],
                  [3, 2, 1]])
    
    # 일반적인 numpy 행렬곱
    start_time = time.time()
    numpy_result = np.dot(A, B)
    numpy_time = time.time() - start_time
    print("Numpy 행렬곱 결과:")
    print(numpy_result)
    print(f"Numpy 수행 시간: {numpy_time:.6f}초")
    
    # 병렬 처리 행렬곱
    start_time = time.time()
    parallel_mm = MatrixMultiplication(A, B)
    parallel_result = parallel_mm.parallel_multiply()
    parallel_time = time.time() - start_time
    print("\n병렬 처리 행렬곱 결과:")
    print(parallel_result)
    print(f"병렬 처리 수행 시간: {parallel_time:.6f}초")
    
    # 결과 검증
    if np.allclose(numpy_result, parallel_result):
        print("\n두 결과가 일치합니다!")
    else:
        print("\n결과가 일치하지 않습니다!")

    # 성능 비교
    print(f"\n성능 비교: {'병렬 처리가 더 빠름' if parallel_time < numpy_time else 'Numpy가 더 빠름'}")

if __name__ == "__main__":
    test_multiplication()
