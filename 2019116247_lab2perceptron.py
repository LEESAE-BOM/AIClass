from sklearn.linear_model import Perceptron
# 샘플과 레이블이다.
X = [[163,43],[160,55],[165,48],[170,80],[175,76],[180,70]]
y = [0, 0, 0, 1,1,1]
# 퍼셉트론을 생성한다. tol는 종료 조건. random_state는 난수의 시드.
clf = Perceptron(tol=1e-3, random_state=0)
# 학습을 수행한다. Stop when previous_loss – loss < tol
clf.fit(X, y)
# 테스트를 수행한다.

print(clf.score(X,y))
print(clf.predict(X))