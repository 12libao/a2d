import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
import scipy
import itertools


class Truss:
    def __init__(self, conn, xpts, E=1e2, A=1.0, bcs=[]):
        self.conn = np.array(conn, dtype=int)
        self.xpts = np.array(xpts, dtype=float)
        self.nelems = len(self.conn)
        self.bcs = bcs

        # Set the number of degrees of freedom
        self.ndof = 2 * (np.max(self.conn) + 1)

        # Compute the dof entries of each element
        self.u0 = np.zeros(self.nelems, dtype=int)
        self.u1 = np.zeros(self.nelems, dtype=int)
        self.v0 = np.zeros(self.nelems, dtype=int)
        self.v1 = np.zeros(self.nelems, dtype=int)
        for i in range(self.nelems):
            self.u0[i] = 2 * self.conn[i][0]
            self.u1[i] = 2 * self.conn[i][1]
            self.v0[i] = 2 * self.conn[i][0] + 1
            self.v1[i] = 2 * self.conn[i][1] + 1

        # Compute the initial length of the bars
        dx = self.xpts[self.u0] - self.xpts[self.u1]
        dy = self.xpts[self.v0] - self.xpts[self.v1]
        self.L0 = np.sqrt(dx**2 + dy**2)

        # Set the values of the stiffnesses
        self.k = E * A / self.L0

        return

    def setBCs(self, u):
        for bc in self.bcs:
            u[bc] = 0.0

        return

    def getEnergy(self, u):
        """
        Get the elastic energy in the elastic system
        """

        # Compute the deformed configuration
        X = self.xpts + u

        # Compute the displacements at each node of each element
        dx = X[self.u0] - X[self.u1]
        dy = X[self.v0] - X[self.v1]

        # Compute the deformed length of each bar
        L = np.sqrt(dx**2 + dy**2)

        # Compute the change in length of the bar
        delta = L - self.L0
        energy = 0.5 * np.dot(self.k, delta**2)

        return energy

    def getR(self, u):
        """
        Get the elastic part of the residual
        """

        # Compute the deformed configuration
        X = self.xpts + u

        # Compute the displacements at each node of each element
        dx = X[self.u0] - X[self.u1]
        dy = X[self.v0] - X[self.v1]

        # Compute the deformed length of each bar
        L = np.sqrt(dx**2 + dy**2)

        # Compute the change in length of the bar
        delta = L - self.L0

        # Compute the residual
        res = np.zeros(self.ndof)
        np.add.at(res, self.u0, self.k * delta * dx / L)
        np.add.at(res, self.u1, -self.k * delta * dx / L)
        np.add.at(res, self.v0, self.k * delta * dy / L)
        np.add.at(res, self.v1, -self.k * delta * dy / L)

        self.setBCs(res)

        return res

    def testR(self, u=None, p=None, dh=1e-6):
        if u is None:
            u = np.random.uniform(size=self.ndof)
        if p is None:
            p = np.random.uniform(size=self.ndof)

        self.setBCs(u)
        self.setBCs(p)

        ans = np.dot(self.getR(u), p)
        fd = (self.getEnergy(u + dh * p) - self.getEnergy(u)) / dh

        print(
            "Energy check: ans = %15.5e  fd = %15.5e  rel. err = %15.5e"
            % (ans, fd, (fd - ans) / ans)
        )
        return

    def getK(self, u):
        """
        Compute the tangent stiffness matrix
        """

        # Compute the deformed configuration
        X = self.xpts + u

        # Compute the displacements at each node of each element
        dx = X[self.u0] - X[self.u1]
        dy = X[self.v0] - X[self.v1]

        # Compute the deformed length of each bar
        L = np.sqrt(dx**2 + dy**2)

        # Compute the change in length of the bar
        delta = L - self.L0

        K = np.zeros((self.ndof, self.ndof))
        dxL = dx / L
        dyL = dy / L
        deltaL = delta / L

        kx = self.k * (dxL * dxL + deltaL * (1.0 - dxL * dxL))
        ky = self.k * (dyL * dyL + deltaL * (1.0 - dyL * dyL))
        kxy = self.k * (dxL * dyL - deltaL * dxL * dyL)

        np.add.at(K, (self.u0, self.u0), kx)
        np.add.at(K, (self.u0, self.u1), -kx)
        np.add.at(K, (self.u0, self.v0), kxy)
        np.add.at(K, (self.u0, self.v1), -kxy)

        np.add.at(K, (self.u1, self.u0), -kx)
        np.add.at(K, (self.u1, self.u1), kx)
        np.add.at(K, (self.u1, self.v0), -kxy)
        np.add.at(K, (self.u1, self.v1), kxy)

        np.add.at(K, (self.v0, self.v0), ky)
        np.add.at(K, (self.v0, self.v1), -ky)
        np.add.at(K, (self.v0, self.u0), kxy)
        np.add.at(K, (self.v0, self.u1), -kxy)

        np.add.at(K, (self.v1, self.v0), -ky)
        np.add.at(K, (self.v1, self.v1), ky)
        np.add.at(K, (self.v1, self.u0), -kxy)
        np.add.at(K, (self.v1, self.u1), kxy)

        # Apply boundary conditions to the matrix
        for bc in self.bcs:
            K[bc, :] = 0.0
            K[:, bc] = 0.0
            K[bc, bc] = 1.0

        return K

    def testK(self, u=None, p=None, dh=1e-6):
        if u is None:
            u = np.random.uniform(size=self.ndof)
        if p is None:
            p = np.random.uniform(size=self.ndof)

        self.setBCs(u)
        self.setBCs(p)

        ans = np.dot(self.getK(u), p)
        fd = (self.getR(u + dh * p) - self.getR(u)) / dh

        for i in range(self.ndof):
            if ans[i] != 0.0:
                print(
                    "res[%3d] = %15.5e  fd[%3d] = %15.5e  rel. err = %15.5e"
                    % (i, ans[i], i, fd[i], (fd[i] - ans[i]) / ans[i])
                )
        return

    def getG(self, u, v):
        """
        Compute: G = d/d(eps) (K(u + eps * v))
        """
        dh = 1e-6
        return (self.getK(u + dh * v) - self.getK(u)) / dh

    def getGprod(self, u, v, p):
        """
        Compute: result = G(u; v) * p
        """

        dh = 1e-6
        return np.dot(self.getG(u, v), p)

    def getHprod(self, u, v, w, p):
        """
        Compute: result = d/d(eps)(G(u + eps * w; v)) * p
        """

        dh = 1e-6
        return (self.getGprod(u + dh * w, v, p) - self.getGprod(u, v, p)) / dh

    def newton(self, f, lam0=1.0, u0=None, tol=1e-8, max_iters=100):
        if u0 is None:
            u0 = np.zeros(self.ndof)

        for i in range(max_iters):
            res = self.getR(u0) - lam0 * f
            rnorm = np.linalg.norm(res)
            print("Full Newton[%3d]  %15.5e" % (i, rnorm))
            if rnorm < tol:
                break
            u0 -= np.linalg.solve(self.getK(u0), res)

        return u0

    def buckling(self, f, lam0, u0):
        K = self.getK(u0)
        u1 = np.linalg.solve(K, f)
        G = self.getG(u0, u1)

        # Solve for the eigenvectors
        eigs, vecs = scipy.linalg.eigh(G, b=K)
        for i in range(len(eigs)):
            if np.fabs(eigs[i]) < 1e-15:
                eigs[i] = -1e-15

        lam = -1.0 / eigs
        indices = np.argsort(lam)
        imin = 0
        for i in range(len(indices)):
            if lam[indices[i]] > 0.0:
                imin = i
                break

        return lam0 + lam[indices[imin:]], vecs[:, indices[imin:]]

    def compute_path(self, f, u0=None, lam0=0.0, lam1=1.0, lam2=0.0, lam3=0.0):
        if u0 is None:
            lam0 = 0.0
            u0 = np.zeros(self.ndof)

        K = self.getK(u0)

        # Solve for u1
        rhs1 = lam1 * f
        u1 = np.linalg.solve(K, rhs1)

        # Solve for u2
        rhs2 = lam2 * f - self.getGprod(u0, u1, u1)
        u2 = np.linalg.solve(K, rhs2)

        # Solve for u3
        rhs3 = (
            lam3 * f - 3.0 * self.getGprod(u0, u1, u2) - self.getHprod(u0, u1, u1, u1)
        )
        u3 = np.linalg.solve(K, rhs3)

        return u0, u1, u2, u3

    def compute_coef(self, u0, V):
        """
        Given the starting point u0 and the subspace V, compute the coefficients
        """

        m = V.shape[-1]

        # Compute the residuals
        res = self.getR(u0)
        r = np.dot(V.T, res)

        # Compute the k coefficients
        J = np.dot(V.T, np.dot(self.getK(u0), V))

        # Compute the g coefficients
        g = np.zeros((m, m, m))
        for k in range(m):
            vk = V[:, k]
            for j in range(m):
                vj = V[:, j]
                g[:, j, k] = np.dot(V.T, self.getGprod(u0, vj, vk))

        # There are a total of  m * (m + 1) * (m + 2) / 6 coefficients
        for i in range(m):
            for j in range(i + 1):
                for k in range(j + 1):
                    perms = itertools.permutations([i, j, k])
                    g0 = 0.0
                    for p in perms:
                        g0 += g[p]
                    g0 /= 6.0

                    for p in perms:
                        g[p] = g0

        # Compute the h coefficients
        h = np.zeros((m, m, m, m))
        for l in range(m):
            vl = V[:, l]
            for k in range(m):
                vk = V[:, k]
                for j in range(m):
                    vj = V[:, j]
                    h[:, j, k, l] = np.dot(V.T, self.getHprod(u0, vj, vk, vl))

        for i in range(m):
            for j in range(i + 1):
                for k in range(j + 1):
                    for l in range(k + 1):
                        perms = itertools.permutations([i, j, k, l])
                        h0 = 0.0
                        for p in perms:
                            h0 += h[p]
                        h0 /= 24.0

                        for p in perms:
                            h[p] = h0

        return r, J, g, h

    def visualize(self, u=None, ax=None, color="b", label=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if u is None:
            X = self.xpts
        else:
            X = self.xpts + u

        for i in range(self.nelems):
            x = [X[self.u0[i]], X[self.u1[i]]]
            y = [X[self.v0[i]], X[self.v1[i]]]
            if i == 0:
                ax.plot(x, y, color=color, marker="o", label=label)
            else:
                ax.plot(x, y, color=color, marker="o")
        ax.axis("equal")

        return


class ROM:
    def __init__(self, r, J, g, h):
        self.ndof = len(r)
        self.r = r
        self.J = J
        self.g = g
        self.h = h

    def setBCs(self, u):
        pass

    def getEnergy(self, u):
        energy = np.dot(u, self.r) + 0.5 * np.dot(u, np.dot(self.J, u))

        for i in range(self.ndof):
            for j in range(self.ndof):
                for k in range(self.ndof):
                    energy += u[i] * u[j] * u[k] * self.g[i, j, k] / 6.0

        for i in range(self.ndof):
            for j in range(self.ndof):
                for k in range(self.ndof):
                    for l in range(self.ndof):
                        energy += u[i] * u[j] * u[k] * u[l] * self.h[i, j, k, l] / 24.0

        return energy

    def getR(self, u, eps=None):
        r = self.r + np.dot(self.J, u)

        for j in range(self.ndof):
            for k in range(self.ndof):
                r[:] += 0.5 * self.g[:, j, k] * u[j] * u[k]

        for i in range(self.ndof):
            for j in range(self.ndof):
                for k in range(self.ndof):
                    for l in range(self.ndof):
                        r[i] += self.h[i, j, k, l] * u[j] * u[k] * u[l] / 24.0
                        r[j] += self.h[i, j, k, l] * u[i] * u[k] * u[l] / 24.0
                        r[k] += self.h[i, j, k, l] * u[i] * u[j] * u[l] / 24.0
                        r[l] += self.h[i, j, k, l] * u[i] * u[j] * u[k] / 24.0

        return r

    def getK(self, u, eps=None):
        J = self.J + np.dot(self.g, u)

        for i in range(self.ndof):
            for j in range(self.ndof):
                for k in range(self.ndof):
                    for l in range(self.ndof):
                        J[i, j] += self.h[i, j, k, l] * u[k] * u[l] / 24.0
                        J[i, k] += self.h[i, j, k, l] * u[j] * u[l] / 24.0
                        J[i, l] += self.h[i, j, k, l] * u[j] * u[k] / 24.0

                        J[j, i] += self.h[i, j, k, l] * u[k] * u[l] / 24.0
                        J[j, k] += self.h[i, j, k, l] * u[i] * u[l] / 24.0
                        J[j, l] += self.h[i, j, k, l] * u[i] * u[k] / 24.0

                        J[k, i] += self.h[i, j, k, l] * u[j] * u[l] / 24.0
                        J[k, j] += self.h[i, j, k, l] * u[i] * u[l] / 24.0
                        J[k, l] += self.h[i, j, k, l] * u[i] * u[j] / 24.0

                        J[l, i] += self.h[i, j, k, l] * u[j] * u[k] / 24.0
                        J[l, j] += self.h[i, j, k, l] * u[i] * u[k] / 24.0
                        J[l, k] += self.h[i, j, k, l] * u[i] * u[j] / 24.0

        # for k in range(self.ndof):
        #     for l in range(self.ndof):
        #         J[:, :] += self.h[:, :, k, l] * u[k] * u[l] / 2.0

        return J

    def testR(self, u=None, p=None, dh=1e-6):
        if u is None:
            u = np.random.uniform(size=self.ndof)
        if p is None:
            p = np.random.uniform(size=self.ndof)

        self.setBCs(u)
        self.setBCs(p)

        ans = np.dot(self.getR(u), p)
        fd = (self.getEnergy(u + dh * p) - self.getEnergy(u)) / dh

        print(
            "Energy check: ans = %15.5e  fd = %15.5e  rel. err = %15.5e"
            % (ans, fd, (fd - ans) / ans)
        )
        return

    def testK(self, u=None, p=None, dh=1e-6):
        if u is None:
            u = np.random.uniform(size=self.ndof)
        if p is None:
            p = np.random.uniform(size=self.ndof)

        self.setBCs(u)
        self.setBCs(p)

        ans = np.dot(self.getK(u), p)
        fd = (self.getR(u + dh * p) - self.getR(u)) / dh

        for i in range(self.ndof):
            if ans[i] != 0.0:
                print(
                    "res[%3d] = %15.5e  fd[%3d] = %15.5e  rel. err = %15.5e"
                    % (i, ans[i], i, fd[i], (fd[i] - ans[i]) / ans[i])
                )
        return

    def newton(self, f, lam0=1.0, u0=None, tol=1e-8, max_iters=100):
        if u0 is None:
            u0 = np.zeros(self.ndof)

        steps = np.linspace(0, 1, 20)[1:]
        for k, step in enumerate(steps):
            lam = step * lam0

            for i in range(max_iters):
                res = self.getR(u0) - lam * f
                rnorm = np.linalg.norm(res)
                print("ROM Newton[%3d]  %15.5e" % (i, rnorm))
                if rnorm < tol:
                    break
                u0 -= np.linalg.solve(self.getK(u0), res)

        return u0


# conn = [[0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 5], [1, 2], [3, 4], [1, 4], [2, 3]]
# xpts = [0, 0, 1, 1, 1, 0.5, 2, 1, 2, 0.5, 3, 0]
# bcs = [0, 1, 10, 11]
# f = [0, 0, 0, -10, 0, 0, 0, -9, 0, 0, 0, 0]

M = 10
nnodes = 2 * (M + 1)
ndof = 2 * nnodes

x = np.linspace(0, 1, M + 1)
y = x * (1.0 - x)

xpts = []
for k in range(M + 1):
    xpts.extend([1.0 * k, y[k], 1.0 * k, 1.0 + y[k]])

conn = [[0, 1]]
for k in range(M):
    conn.extend(
        [
            [2 * k, 2 * (k + 1)],
            [2 * k + 1, 2 * (k + 1) + 1],
            [2 * (k + 1), 2 * (k + 1) + 1],
            [2 * k, 2 * (k + 1) + 1],
            [2 * k + 1, 2 * (k + 1)],
        ]
    )

bcs = [0, 1, 2 * (2 * M + 1), 2 * (2 * M + 1) + 1]

truss = Truss(conn, xpts, bcs=bcs)
truss.testR()
truss.testK()

f = np.zeros(ndof)
f[3::4] = -0.1
truss.setBCs(f)

# Solve for the solution at lam0
lam0 = 0.2
u0 = truss.newton(f, lam0)

# Compute the solution path
u0, u1, u2, u3 = truss.compute_path(f, u0=u0, lam0=lam0)

# Compute the buckling modes about lam0
lamb, vec = truss.buckling(f, lam0, u0)

# Compute the second-order correction for w0
K = truss.getK(u0)
u1 = np.linalg.solve(K, f)
uc = u0 + (lamb[0] - lam0) * u1
Kc = truss.getK(uc)

v0 = vec[:, 0]
w0 = -np.linalg.solve(Kc, truss.getGprod(uc, v0, v0))
w0 = w0 - np.dot(v0, np.dot(Kc, w0)) * v0
w0 = w0 / np.linalg.norm(w0)

# Construct a ROM using the specified coefficients
V = np.zeros((truss.ndof, 4))
V[:, 0] = u1
V[:, 1] = u2
V[:, 2] = u3
V[:, 3] = v0
# V[:, 4] = w0

r, J, g, h = truss.compute_coef(u0, V)
rom = ROM(r, J, g, h)
p = np.zeros(rom.ndof)
rom.testR()
rom.testK()

lam = 3.0
alpha = rom.newton(np.dot(V.T, f), lam0=lam)

rom.testR(u=alpha)
rom.testK(u=alpha)

ur = u0 + np.dot(V, alpha)
uf = truss.newton(f, lam0=lam)

# lam = lam0 + s
s = lam - lam0
ue = u0 + s * u1 + 0.5 * s**2 * u2 + 1.0 / 6.0 * s**3 * u3

fig, ax = plt.subplots(1, 1)
truss.visualize(ax=ax, color="Black", label="initial")
truss.visualize(ur, ax, color="Red", label="ROM")
truss.visualize(uf, ax, color="Blue", label="exact")
truss.visualize(ue, ax, color="Green", label="extrapolation")
ax.legend()

fig, ax = plt.subplots(1, 1)
truss.visualize(ax=ax, color="Black", label="initial")
truss.visualize(v0, ax, color="Red", label="buckling mode")
truss.visualize(w0, ax, color="Blue", label="second-order mode")
ax.legend()

cmap = mpl.colormaps["coolwarm"]

xi = np.linspace(0, lam - lam0, 10)
fig, ax = plt.subplots(1, 1)
for i, s in enumerate(xi):
    lam = lam0 + s
    u = u0 + s * u1 + 0.5 * s**2 * u2 + 1.0 / 6.0 * s**3 * u3

    res = truss.getR(u) - lam * f
    print("norm[%3d] = %15.5e" % (i, np.linalg.norm(res)))

    truss.visualize(u, ax, color=cmap(s / max(xi)))

plt.show()
