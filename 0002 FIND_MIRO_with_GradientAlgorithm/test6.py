# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import func as f
# 초기상태의 미로 모습

# 전체 그림의 크기 및 그림을 나타내는 변수 선언
fig = plt.figure(figsize=(5,5))
ax = plt.gca()


#붉은 벽 그리기
plt.plot([1,1],[0,1],color = 'red', linewidth = 2)
plt.plot([1,2],[2,2],color = 'red', linewidth = 2)
plt.plot([2,2],[2,1],color = 'red', linewidth = 2)
plt.plot([2,3],[1,1],color = 'red', linewidth = 2)

#상태를 의미하는 문자열(S0-S8)표시
plt.text(0.5,2.5,'S0',size=14,ha='center')
plt.text(1.5,2.5,'S1',size=14,ha='center')
plt.text(2.5,2.5,'S2',size=14,ha='center')
plt.text(0.5,1.5,'S3',size=14,ha='center')
plt.text(1.5,1.5,'S4',size=14,ha='center')
plt.text(2.5,1.5,'S5',size=14,ha='center')
plt.text(0.5,0.5,'S6',size=14,ha='center')
plt.text(1.5,0.5,'S7',size=14,ha='center')
plt.text(2.5,0.5,'S8',size=14,ha='center')
plt.text(0.5,2.3,'START',ha='center')
plt.text(2.5,0.3,'GOAL',ha='center')

ax.set_xlim(0,3)
ax.set_ylim(0,3)

plt.tick_params(axis = 'both', which = 'both',bottom=False,top=False,labelbottom=False,right=False,left=False,labelleft=False)

#S0에 녹샌 원으로 현재 위치를 표시
line, = ax.plot([0.5],[2.5],marker="o",color='g',markersize=60)

# 정책을 결정하는 파라미터의 초깃값 theta_0을 설정

# 줄은 상태 0~7, 열은 행동 방향(상 , 우 , 하, 좌 순)--> 시계 방향
theta_0 = np.array([[np.nan,1,     1,     np.nan], #s0
                    [np.nan,1,     np.nan,1],      #s1
                    [np.nan,np.nan,1,     1],      #s2     
                    [1,     1,     1,     np.nan], #s3
                    [np.nan,np.nan,1,     1], #s4
                    [1,     np.nan,np.nan,np.nan], #s5
                    [1,     np.nan,np.nan,np.nan], #s6
                    [1,     1,     np.nan,np.nan], #s7
                    ]) #s8은 목표지점이므로 정책이 없다.


# 1. 정책 파라미터 theta_0를 행동정책 pi_0로 변환
pi_0 = f.softmax_convert_into_pi_from_theta(theta_0)

print(pi_0)

stop_epsilon = 10**-4 # 정책의 변화가 10^-4보다 작아지면 학습 종료

theta = theta_0
pi = pi_0

is_continue = True

count = 1

while is_continue: # False 가 될 때까지 반복
    s_a_history = f.goal_maze_ret_s_a(pi) # 정책 pi를 따라 미로를 탐색한 히스토리를 구함
    print(s_a_history)
    new_theta = f.update_theta(theta,pi,s_a_history) #파라미터 theta를 수정
    print(new_theta)
    new_pi = f.softmax_convert_into_pi_from_theta(new_theta) # 정책 pi를 수정
    print(new_pi)
    print(np.sum(np.abs(new_pi-pi))) # 정책의 변화를 출력
    print("목표 지점에 이르기까지 걸린 단계 수는 " +str(len(s_a_history)-1)+"단계입니다.")

    if np.sum(np.abs(new_pi-pi)) < stop_epsilon:
        is_continue = False
    else:
        theta = new_theta
        pi=new_pi
    is_continue = False

'''
np.set_printoptions(precision=3, suppress= True) # 유효 자리수 3, 지수는 표시하지 않도록 설정
print(pi)



# 에이전트의 이동 과정을 시각화
# 참고 ~~ http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-notebooks/
from matplotlib import animation
#from IPython.display import HTML
def init():
    #배경 이미지 초기화
    line.set_data([],[])
    return (line,)

def animate(i):
    #프레임 단위로 이미지 생성
    state = s_a_history[i][0] #현재 위치
    x = (state % 3) + 0.5 # 상태의 x 좌표 : 3으로 나눈 나머지 +0.5
    y = 2.5 -int(state/3) # y 좌표 : 2.5에서 3으로 나눈 몫을 뺌
    line.set_data(x,y)
    return (line,)

anim = animation.FuncAnimation(fig,animate,init_func=init,frames=len(s_a_history),interval =200, repeat = False)
plt.show()
#HTML(anim.to_jshtml())

'''