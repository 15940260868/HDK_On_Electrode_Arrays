

def test_ref(x):
    x[0] = 100

i = [.0]
test_ref(i)
print(i[0])
