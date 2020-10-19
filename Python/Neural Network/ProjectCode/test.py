
import numpy as mat


a = mat.arange(9).reshape(3, 3)
a = mat.where(a < 4, 0, a)
l = mat.where(a > 0)
print(mat.log(a[mat.where(a > 0)]))
a[l] = mat.log(a[mat.where(a > 0)])
print(a)
