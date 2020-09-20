def findMaxSubArray(A):
    s = 0
    pos = -1
    max_s = -99999999999
    for i in range(len(A)):
        s += A[i]
        if s > max_s:
            max_s = s
            start = pos + 1
            end = i
        if s < 0:
            s = 0
            pos = i

    end += 1
    if start == end:
        return A[start], A[start]
    else:
        return A[start:end], max_s


a = [-2, -1, 5, -4, 5, -2, -1, -5, -4]
print(findMaxSubArray(a))