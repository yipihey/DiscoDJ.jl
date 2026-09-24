"""Background cosmology, EH98 linear P(k), exact 1st/2nd-order growth and
leapfrog kick/drift integrals for flat LCDM.

Conventions (match DiscoDJNative / DISCO-DJ):
  lengths Mpc/h, time in 1/H0, k in h/Mpc, P(k) in (Mpc/h)^3.
  x(q,a) = q + D1(a) psi1(q) + D2(a) psi2(q),   D1 -> a, D2 -> -3/7 a^2 at early times.
"""
import numpy as np
from scipy.integrate import solve_ivp, quad


class Cosmology:
    # Planck18EEBAOSN, the DiscoDJNative default
    def __init__(self, Omega_c=0.259622, Omega_b=0.0488911, h=0.67742,
                 sigma8=0.8105, n_s=0.96822, Tcmb=2.72548):
        self.Omega_c, self.Omega_b, self.h = Omega_c, Omega_b, h
        self.sigma8, self.n_s, self.Tcmb = sigma8, n_s, Tcmb
        self.Om = Omega_c + Omega_b
        self.OL = 1.0 - self.Om
        self._growth_tables()
        self._pk_norm()

    # ---------------- background ----------------
    def E(self, a):
        return np.sqrt(self.Om * a**-3 + self.OL)

    def dlnE_da(self, a):
        return -1.5 * self.Om * a**-4 / self.E(a)**2

    # ---------------- growth ----------------
    def _growth_tables(self):
        """Exact LCDM growth factors up to third order (DISCO-DJ conventions):
        D1, D2 (~ -3/7 D1^2), D3a (~ 1/3 D1^3), D3b (~ -10/21 D1^3) and the
        transverse D3c (~ 1/7 D1^3), each normalised by D1(a=1)^n.

          L[D] = D'' + (3/a + E'/E) D' - src D,  src = 1.5 Om / (a^5 E^2)
          L[D2]  = -src D1^2
          L[D3a] =  2 src D1^3
          L[D3b] =  2 src (D1 D2 - D1^3)
          dD3c/da = D2 D1' - D1 D2'
        """
        a0 = 1e-3
        def rhs(a, y):
            D1, dD1, D2, dD2, D3a, dD3a, D3b, dD3b, D3c = y
            fric = 3.0 / a + self.dlnE_da(a)
            src = 1.5 * self.Om / (a**5 * self.E(a)**2)
            return [dD1, -fric * dD1 + src * D1,
                    dD2, -fric * dD2 + src * D2 - src * D1**2,
                    dD3a, -fric * dD3a + src * D3a + 2 * src * D1**3,
                    dD3b, -fric * dD3b + src * D3b + 2 * src * (D1 * D2 - D1**3),
                    D2 * dD1 - D1 * dD2]
        # EdS growing modes at a0 (radiation ignored, consistent with E(a))
        y0 = [a0, 1.0, -3/7 * a0**2, -6/7 * a0,
              1/3 * a0**3, a0**2, -10/21 * a0**3, -10/7 * a0**2, 1/7 * a0**3]
        ag = np.geomspace(a0, 1.0, 4000)
        s = solve_ivp(rhs, (a0, 1.0), y0, t_eval=ag, rtol=1e-11, atol=1e-16,
                      method="DOP853")
        self._a = ag
        D1_1 = s.y[0, -1]
        # normalise so that D1(a=1) = 1 (P(k) below is the a=1 spectrum)
        self._D1, self._dD1 = s.y[0] / D1_1, s.y[1] / D1_1
        self._D2, self._dD2 = s.y[2] / D1_1**2, s.y[3] / D1_1**2
        self._D3a, self._D3b, self._D3c = (s.y[4] / D1_1**3, s.y[6] / D1_1**3,
                                           s.y[8] / D1_1**3)

    def D3(self, a):
        """(D3a, D3b, D3c) at a."""
        return tuple(np.interp(a, self._a, t) for t in (self._D3a, self._D3b, self._D3c))

    def D1(self, a):
        return np.interp(a, self._a, self._D1)

    def D2(self, a):
        return np.interp(a, self._a, self._D2)

    def f1(self, a):
        return a * np.interp(a, self._a, self._dD1) / self.D1(a)

    def f2(self, a):
        return a * np.interp(a, self._a, self._dD2) / self.D2(a)

    # ---------------- leapfrog factors ----------------
    # dx/da = p / (a^3 E),  dp/da = -grad(Phi) / (a E),  lap Phi = 1.5 Om delta / a
    def drift_factor(self, a0, a1):
        return quad(lambda a: 1.0 / (a**3 * self.E(a)), a0, a1, epsrel=1e-12)[0]

    def kick_factor(self, a0, a1):
        return quad(lambda a: 1.0 / (a * self.E(a)), a0, a1, epsrel=1e-12)[0]

    # ---------------- linear P(k) ----------------
    def transfer_EH(self, k):
        """Eisenstein & Hu (1998) with baryons; port of DiscoDJNative.eisenstein_hu."""
        k = np.asarray(k, dtype=np.float64)
        th = self.Tcmb / 2.7
        th2, th4 = th**2, th**4
        omh2 = self.Om * self.h**2
        f_b = self.Omega_b / self.Om
        obh2 = omh2 * f_b
        z_eq = 2.50e4 * omh2 / th4
        k_eq = 0.0746 * omh2 / th2 / self.h
        z_d1 = 0.313 * omh2**(-0.419) * (1 + 0.607 * omh2**0.674)
        z_d2 = 0.238 * omh2**0.223
        z_d = 1291.0 * omh2**0.251 / (1 + 0.659 * omh2**0.828) * (1 + z_d1 * obh2**z_d2)
        R_d = 31.5 * obh2 / th4 * (1000 / (1 + z_d))
        R_eq = 31.5 * obh2 / th4 * (1000 / z_eq)
        s = 2 / 3 / k_eq * np.sqrt(6 / R_eq) * np.log(
            (np.sqrt(1 + R_d) + np.sqrt(R_d + R_eq)) / (1 + np.sqrt(R_eq)))
        k_silk = 1.6 * obh2**0.52 * omh2**0.73 * (1 + (10.4 * omh2)**(-0.95)) / self.h
        a1 = (46.9 * omh2)**0.670 * (1 + (32.1 * omh2)**(-0.532))
        a2 = (12.0 * omh2)**0.424 * (1 + (45.0 * omh2)**(-0.582))
        al_c = a1**(-f_b) * a2**(-f_b**3)
        b1 = 0.944 / (1 + (458 * omh2)**(-0.708))
        b2 = (0.395 * omh2)**(-0.0266)
        be_c = 1 / (1 + b1 * ((1 - f_b)**b2 - 1))
        yG = (1 + z_eq) / (1 + z_d)
        Gy = yG * (-6 * np.sqrt(1 + yG) + (2 + 3 * yG) *
                   np.log((np.sqrt(1 + yG) + 1) / (np.sqrt(1 + yG) - 1)))
        al_b = 2.07 * k_eq * s * (1 + R_d)**(-0.75) * Gy
        be_b = 0.5 + f_b + (3 - 2 * f_b) * np.sqrt((17.2 * omh2)**2 + 1)
        be_node = 8.41 * omh2**0.435

        q = k / (13.41 * k_eq)
        ks = k * s
        Cf = lambda ac: 14.2 / ac + 386 / (1 + 69.9 * q**1.08)
        lt = lambda b: np.log(np.e + 1.8 * b * q)
        T0t = lambda ac, bc: lt(bc) / (lt(bc) + Cf(ac) * q**2)
        f = 1 / (1 + (ks / 5.4)**4)
        T_c = f * T0t(1.0, be_c) + (1 - f) * T0t(al_c, be_c)
        s_t = s / (1 + (be_node / ks)**3)**(1 / 3)
        Tb1 = T0t(1.0, 1.0) / (1 + (ks / 5.2)**2)
        Tb2 = al_b / (1 + (be_b / ks)**3) * np.exp(-(k / k_silk)**1.4)
        T_b = np.sinc(k * s_t / np.pi) * (Tb1 + Tb2)
        return f_b * T_b + (1 - f_b) * T_c

    def _pk_norm(self):
        self._A = 1.0
        self._A = self.sigma8**2 / self.sigma2_TH(8.0)

    def Pk(self, k):
        """Linear P(k) at a=1 (D1(1) normalised: sigma8 refers to a=1)."""
        k = np.asarray(k, dtype=np.float64)
        out = np.zeros_like(k)
        m = k > 0
        out[m] = self._A * k[m]**self.n_s * self.transfer_EH(k[m])**2
        return out

    def sigma2_TH(self, R, kmin=1e-5, kmax=1e3, n=20000, extra_filter_R=None):
        k = np.geomspace(kmin, kmax, n)
        W = W_TH(k * R)
        if extra_filter_R is not None:
            W = W * W_TH(k * extra_filter_R)
        integ = k**3 * self.Pk(k) * W**2 / (2 * np.pi**2)
        return np.trapezoid(integ, np.log(k))


def W_TH(x):
    """Fourier transform of a real-space spherical top hat."""
    x = np.asarray(x, dtype=np.float64)
    out = np.ones_like(x)
    m = x > 1e-4
    xm = x[m]
    out[m] = 3 * (np.sin(xm) - xm * np.cos(xm)) / xm**3
    s = ~m
    out[s] = 1 - x[s]**2 / 10
    return out
