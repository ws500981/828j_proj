import re
import os

# to create sample_i.txt, remove all rubbish stuff.
f = open("/content/sample2.txt")
a = f.readlines()
f.close()

# add more quantifiers
f = open("/content/result.txt","w")
for i in range(len(a)):
  t = a[i].split("?")[0]
  if "not" in t:
    f.write(t+"\n")
    for j in ["any", "most", "some", "few", "each", "one"]:
      t1 = t.replace("not", j)
      f.write(t1+"\n")
  else:
    f.write(t+"\n")
f.close()

f = open("/content/result.txt")
a = f.readlines()
f.close()

# create a mixture
def check(lst, sent):
  for i in lst:
    if i in sent:
      return True
  return False

def count(lst, sent):
  c = 0
  for i in lst:
    if i in sent:
      c += len(re.findall(i, sent))
  return c

def present(lst, sent):
  c = []
  for i in lst:
    if i in sent:
      c.append(i)
  return c

f = open("/content/fin_result1.txt","w")
l = ["not", "any", "most", "some", "few", "each", "one"]
for i in range(len(a)):
  if (check(l, a[i]) == True) and (count(l, a[i]) == 1):
    f.write(a[i])
    c = present(l, a[i])
    temp = a[i].split()
    t1 = temp[temp.index(c[0]) + 1]
    if t1 == "{A}":
      for j in l:
        if j != c[0]:
          ts = a[i].split(",")[0]
          te = a[i].split(",")[1]
          t = ts + "," + te.replace("{B}",j+" {B}")
          f.write(t)
    elif t1 == "{B}":
      for j in l:
        if j != c[0]:
          ts = a[i].split(",")[0]
          te = a[i].split(",")[1]
          t = ts + "," + te.replace("{A}",j+" {A}")
          f.write(t)
  elif (check(l, a[i]) == True) and (count(l, a[i]) == 2):
    f.write(a[i])
    c = present(l, a[i])
    for j in c:
      for k in l:
        if k != j:
          ts = a[i].split(",")[0]
          te = a[i].split(",")[1]
          if j+" {A}" in te:
            t = ts + "," + te.replace(j+" {A}", k+" {A}")
          elif j+" {B}" in te:
            t = ts + "," + te.replace(j+" {B}", k+" {B}")
          f.write(t)
  else:
    f.write(a[i])
f.close()

f = open("/content/fin_result1.txt","r")
a = f.readlines()
print(len(a))
a = list(set(a))
print(len(a))
f.close()