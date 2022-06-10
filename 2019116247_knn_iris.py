from sklearn.datasets import load_iris
iris = load_iris() # 샘플데이터 로드
print(iris)
print('\n')# iris 객체 구조 확인
print(iris.data)
print('\n')
print(iris.feature_names)
print('\n')
# 정수는 꽃의 종류를 나타낸다.: 0 = setosa, 1=versicolor, 2=virginica
print(iris.target)
print('\n')
from sklearn.model_selection import train_test_split
X = iris.data
y = iris.target
# (80:20)으로 분할한다.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
print(X_train.shape)
print(X_test.shape)
print('\n')
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics  # 성능측정 관련 함수들을 포함하는 모듈
#학습 단계
knn = KNeighborsClassifier(n_neighbors=6)#weights = ‘uniform’ or ‘distance’
knn.fit(X_train, y_train)
# 테스트 단계
y_pred = knn.predict(X_test)
# 예측 정확도 점수 출력
scores = metrics.accuracy_score(y_test, y_pred)
print(scores)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y) # 사용 가능한 모든 데이터를 이용하여 학습
# 0 = setosa, 1=versicolor, 2=virginica
classes = {0:'setosa',1:'versicolor', 2:'virginica'}  # dictionary= {key: value}
# 아직 보지 못한 새로운 데이터를 제시해보자.
x_new = [[3,4,5,2],[5,4,2,2]]
y_predict = knn.predict(x_new)
print(y_predict)

print(classes[y_predict[0]])
print(classes[y_predict[1]])