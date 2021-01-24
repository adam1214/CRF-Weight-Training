#-*-coding:utf-8-*-
def forward(N,M,A,B,P,observed):
    p = 0.0
    #观察到的状态数目
    LEN = len(observed)
    #中间概率LEN*M
    Q = [([0]*N) for i in range(LEN)] # [[0, 0, 0], [0, 0, 0]]
    #第一个观察到的状态,状态的初始概率乘上隐藏状态到观察状态的条件概率。
    for j in range(N):
        Q[0][j] = P[j]*B[j][observation.index(observed[0])]
    # Q = [[0.03, 0.06, 0.12], [0, 0, 0]]
    #第一个之后的状态，首先从前一天的每个状态，转移到当前状态的概率求和，然后乘上隐藏状态到观察状态的条件概率。
    for i in range(1,LEN):
        for j in range(N):
            sum = 0.0
            for k in range(N):
                print(Q[i-1][k])
                print(A[k][j])
                sum += Q[i-1][k]*A[k][j]
            print(B[j][observation.index(observed[i])])
            Q[i][j] = sum * B[j][observation.index(observed[i])]
    print(Q)
    for i in range(N):
        print(Q[LEN-1][i])
        p += Q[LEN-1][i]
    return p

if __name__ == "__main__":
    
    # 3种隐藏层状态:sun cloud rain
    hidden = []
    hidden.append('sun')
    hidden.append('cloud')
    hidden.append('rain')
    N = len(hidden)
    # 3种观察层状态:dry damp soggy
    observation = []
    observation.append('dry')
    observation.append('damp')
    observation.append('soggy')
    M = len(observation)
    # 初始状态矩阵（1*N第一天是sun，cloud，rain的概率）
    P = (0.3,0.3,0.4)
    # 状态转移矩阵A（N*N 隐藏层状态之间互相转变的概率）
    A=((0.2,0.3,0.5),(0.1,0.5,0.4),(0.6,0.1,0.3))
    # 混淆矩阵B（N*M 隐藏层状态对应的观察层状态的概率）
    B=((0.1,0.5,0.4),(0.2,0.4,0.4),(0.3,0.6,0.1))
    #假设观察到一组序列为observed，输出HMM模型（N，M，A，B，P）产生观察序列observed的概率
    # observed = ['dry']
    # print(forward(N,M,A,B,P,observed))
    # observed = ['damp']
    # print(forward(N,M,A,B,P,observed))
    observed = ['dry','damp']
    print(forward(N,M,A,B,P,observed))
    # observed = ['dry','damp','soggy']
    # print(forward(N,M,A,B,P,observed))
    