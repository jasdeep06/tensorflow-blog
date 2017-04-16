import random,string
def generate_string(size=6, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
def processed_string(size,suff):
    s=generate_string(size)
    mutable=list(s)
    for i in range(0,int(size/5)):
        mutable[5*i-1]=suff
    return "".join(mutable)
print(processed_string(100,"3"))
