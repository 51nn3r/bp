MOD = 1_000_000_007

def is_lucky(x: int) -> bool:
    if x == 0:
        return False
    while x:
        d = x % 10
        if d != 2 and d != 7:
            return False
        x //= 10
    return True


def mod_pow(a: int, e: int, m: int = MOD) -> int:
    res = 1
    while e:
        if e & 1:
            res = res * a % m
        a = a * a % m
        e >>= 1
    return res


data = list(map(int, input().strip().split()))
if len(data) < 2:
    raise  "Chýba n alebo k"

n, k = data[:2]
arr = data[2:]
assert len(arr) == n

cnt = {}
U = 0

for x in arr:
    if is_lucky(x):
        cnt[x] = cnt.get(x, 0) + 1
    else:
        U += 1

m = len(cnt)
k = min(k, n)

limit = min(k, m)
dp = [0] * (limit + 1)
dp[0] = 1
for c in cnt.values():
    for j in range(limit - 1, -1, -1):
        if dp[j]:
            dp[j + 1] = (dp[j + 1] + dp[j] * c) % MOD

fact = [1] * (n + 1)
for i in range(1, n + 1):
    fact[i] = fact[i - 1] * i % MOD

inv_fact = [1] * (n + 1)
inv_fact[n] = mod_pow(fact[n], MOD - 2)
for i in range(n, 0, -1):
    inv_fact[i - 1] = inv_fact[i] * i % MOD


def nCr(n_: int, r_: int) -> int:
    if r_ < 0 or r_ > n_:
        return 0
    return fact[n_] * inv_fact[r_] % MOD * inv_fact[n_ - r_] % MOD


ans = 0
for t in range(0, limit + 1):
    ways_lucky = dp[t]
    ways_normal = nCr(U, k - t)
    ans = (ans + ways_lucky * ways_normal) % MOD

print(ans)
