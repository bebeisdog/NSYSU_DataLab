import numpy as np
# import cupy as np
def input_matrix():
    '''
    轉換成矩陣形式
    '''
    row = int(input("輸入矩陣有幾列:"))
    column = int(input("輸入矩陣有幾行:"))
    print("Enter the entries in a single line (separated by space): ")

    entries = list(map(float, input().split()))
    matrix = np.array(entries).reshape(row, column)
    print('你輸入的矩陣')
    print(matrix)
    return matrix

def rank(matrix):
    temp = []
    for i in range(matrix.shape[0]): # 行
        count = 0
        for j in range(matrix.shape[1]): # 列
            if matrix[i, j] != 0: # 找出誰最先出現非零數
                break
            count += 1 
        print('第 {} 列，第 {} 行，出現非零數, 前面有 {} 個 0'.format(i, j, count))
        temp.append((i, j, count))

    answer = []
    final = sorted(temp, key = lambda x: x[1])
    for i in range(len(final)):
        answer.append(final[i][0])
    # print(answer)
    return np.array(answer), temp

if __name__ == '__main__':
    calculate = {}
    print('可輸入五個矩陣:')

    for i in range(1,6):
        matrix = input_matrix()
        answer, temp = rank(matrix)
        print(answer)

        test2 = {}
        for j in range(len(temp)):
            test2['{}'.format(j)] = temp[j][-1]
        calculate['B{}'.format(i+1)] = test2

    print(calculate)
