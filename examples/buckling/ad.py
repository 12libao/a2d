from icecream import ic
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import scipy


# generate a random matrix
def rand_symm_mat(n=10, eig_low=0.1, eig_high=100.0, nrepeat=1):
    # Randomly generated matrix that will be used to generate the eigenvectors
    QRmat = -1.0 + 2 * np.random.uniform(size=(n, n))

    Q, _ = np.linalg.qr(QRmat, mode="complete")  # Construct Q via a Q-R decomposition

    if nrepeat == 1:
        lam = np.random.uniform(low=eig_low, high=eig_high, size=n)
    else:
        lam = np.hstack(
            (
                eig_low * np.ones(nrepeat),
                np.random.uniform(low=eig_low, high=eig_high, size=n - nrepeat),
            )
        )

    return np.dot(Q, np.dot(np.diag(lam), Q.T))  # Compute A = Q*Lambda*Q^{T}


# set the size of the matrix
N = 10
eps = 1e-30
epsi = 1 / eps

# set the state of the random generator to zero
np.random.seed(12345)

A = rand_symm_mat(n=N, eig_low=-1.0, eig_high=1.0, nrepeat=1)
B = np.random.uniform(low=-1.0, high=1.0, size=(N, N))
dA = np.random.uniform(low=-1.0, high=1.0, size=(N, N))
dB = np.random.uniform(low=-1.0, high=1.0, size=(N, N))
bC = np.random.uniform(low=-1.0, high=1.0, size=(N, N))

Ae = A + dA * eps * 1j
Be = B + dB * eps * 1j
Ce = Ae + Be
C = np.real(Ce)

dC = dA + dB
bA = bC
bB = bC

cvt_error = np.linalg.norm(dC - epsi * np.imag(Ce))
adjoint_error = np.linalg.norm(dA.T @ bA + dB.T @ bB - dC.T @ bC)
ic('Addition')
ic(cvt_error)
ic(adjoint_error)
