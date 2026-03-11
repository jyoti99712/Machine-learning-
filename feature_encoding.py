from sklearn.preprocessing import OneHotEncoder

  color = [["red"],["green"],["blue"]]

  encoder=OneHotEncoder()

  result=encoder.fit_transform(color)

  print(result.toarray())