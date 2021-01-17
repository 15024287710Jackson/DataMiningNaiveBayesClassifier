#求前20项的斐波那契数
a = 0
b = 1
for _ in range(20):
    (a, b) = (b, a + b)
    # print(a, end=' ')
    print(2)
