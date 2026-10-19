"""
Piezo-transducer stroke from hand-scanned Michelson calibration traces.

For every trace (file idx, channel ch) this:
  1. Lock-in demodulates the RAW signal at the drive frequency -> signed 109 kHz amplitude q(t)
     (the drive frequency is refined automatically, and the orthogonal quadrature p(t) is pure noise)
  2. Low-passes the RAW signal with the same bandwidth           -> slow fringe signal s(t)
  3. Gets the fringe centre C and half-amplitude Bs from s(t)
  4. Uses sin^2(phi) = 1 - ((s - C)/Bs)^2 and q = S*sin(phi) + noise to fit S by least squares,
     removing the noise bias with the variance of p(t)
  5. Solves R = S / (G*Bs) = 2*J1(beta)/J0(beta) for beta, then d0 = beta*lambda/(4*pi)

Because S and Bs come from the same trace, fringe-contrast drift between traces cancels.
Partial / irregular hand sweeps are fine as long as the trace reaches both fringe extremes.

IMPORTANT: run this on unfiltered data. Your current .apply(..., inplace=True) calls overwrite
the raw signal, so restart the kernel / reload `cals` before running this.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.special import j0, j1
from scipy.optimize import brentq

# ---------------- settings ----------------
LAMBDA     = 1064e-9   # laser wavelength (m)
F_DRIVE    = 109e3      # nominal drive frequency (Hz); refined per trace
FS_OUT     = 50e3       # sample rate after decimation (Hz); keep 2*F_DRIVE aliasing far from LP_CUT
LP_CUT     = 2000.0     # bandwidth of slow and demodulated signals (Hz); must exceed the fastest fringe rate
TRIM       = 0.005      # s discarded at each end (filter transients)
G          = 1          # gain of 109 kHz path relative to DC. MEASURE THIS (detector roll-off etc.)
SIN2_MIN   = 0.3        # only use points with sin^2(phi) above this in the S fit
MIN_SWEEPS = 2          # trace must cross a full half-fringe at least this many times
MAX_DF     = 100.0      # max |actual - nominal| drive frequency searched (Hz)
MIN_SNR    = 2.0        # S must exceed the demodulated noise rms by this factor
MAX_EXT    = 0.4        # at the fringe extremes the 109 kHz signal must (nearly) vanish


def _lowpass(y, fs):
    sos = butter(4, LP_CUT, btype="low", fs=fs, output="sos")
    if np.iscomplexobj(y):
        return sosfiltfilt(sos, y.real) + 1j * sosfiltfilt(sos, y.imag)
    return sosfiltfilt(sos, y)


def beta_from_ratio(R):
    """Invert R = 2 J1(beta) / J0(beta) on 0 < beta < 2.405 (monotonic there)."""
    if R <= 0:
        return 0.0
    return brentq(lambda b: 2 * j1(b) / j0(b) - R, 1e-12, 2.4)


def analyze(t, x):
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    fs = 1.0 / np.median(np.diff(t))
    n = max(1, int(round(fs / FS_OUT)))
    m = (len(x) // n) * n
    x, t = x[:m], t[:m]
    fs_d = fs / n

    # 1. lock-in at the nominal drive frequency, boxcar-decimate, then low-pass
    z = (x * np.exp(-2j * np.pi * F_DRIVE * t)).reshape(-1, n).mean(axis=1)
    s = x.reshape(-1, n).mean(axis=1)
    td = t.reshape(-1, n).mean(axis=1)
    z = 2 * _lowpass(z, fs_d)          # complex amplitude of the 109 kHz component
    s = _lowpass(s, fs_d)

    keep = (td > td[0] + TRIM) & (td < td[-1] - TRIM)
    z, s, td = z[keep], s[keep], td[keep]

    # refine drive frequency: z^2 removes the sign flips of sin(phi), leaving a pure rotation
    # (spectral peak of z^2 sits at 2*df; zero-padded FFT is robust to noise)
    w = (z ** 2) * np.hanning(z.size)
    nfft = 16 * (1 << int(np.ceil(np.log2(z.size))))
    W = np.fft.fft(w, nfft)
    fw = np.fft.fftfreq(nfft, 1 / fs_d)
    band = np.abs(fw) < 2 * MAX_DF
    df = fw[band][np.argmax(np.abs(W[band]))] / 2
    z = z * np.exp(-2j * np.pi * df * (td - td[0]))
    theta0 = 0.5 * np.angle(np.sum(z ** 2))
    zr = z * np.exp(-1j * theta0)
    q, p = zr.real, zr.imag            # q = S sin(phi) + noise, p = noise only

    # 3. fringe centre and half-amplitude
    lo, hi = np.percentile(s, [0.1, 99.9])
    C, Bs = (hi + lo) / 2, (hi - lo) / 2
    u = (s - C) / Bs                   # = cos(phi)

    # count full half-fringe sweeps (crossing from one extreme to the other)
    state = np.where(u > 0.8, 1, np.where(u < -0.8, -1, 0))
    nz = state[state != 0]
    sweeps = int(np.count_nonzero(np.diff(nz) != 0)) if nz.size else 0

    # 4. fit S with noise-bias removal
    s2 = np.clip(1 - u ** 2, 0, 1)
    sel = s2 > SIN2_MIN
    noise_var = np.mean(p[sel] ** 2)
    S2 = np.sum((q[sel] ** 2 - noise_var) * s2[sel]) / np.sum(s2[sel] ** 2)
    S = np.sqrt(max(S2, 0.0))
    corr = np.corrcoef(q[sel] ** 2, s2[sel])[0, 1] if sel.sum() > 10 else np.nan

    # 5. displacement
    R = S / (G * Bs)
    beta = beta_from_ratio(R)
    d0 = beta * LAMBDA / (4 * np.pi)

    # check that the slow-signal extremes are real fringe extremes (sin(phi) -> 0 there),
    # not hand reversals partway through a fringe, which would give a wrong Bs
    ext = np.abs(u) > 0.95
    ext_ratio = (np.sqrt(max(np.mean(q[ext] ** 2) - noise_var, 0.0)) / S
                 if (ext.sum() > 10 and S > 0) else np.inf)
    snr = S / np.sqrt(noise_var)
    usable = (sweeps >= MIN_SWEEPS) and (snr >= MIN_SNR) and (ext_ratio <= MAX_EXT)
    return dict(td=td, env=np.abs(z), s=s, u=u, q=q, S=S, B=Bs, C=C, R=R, beta=beta,
                d0=d0, stroke_pp=2 * d0, sweeps=sweeps, corr=corr,
                noise=np.sqrt(noise_var), snr=snr, ext_ratio=ext_ratio, df=df, usable=usable)


# ---------------- driver for your `cals` object ----------------
def load(cals, idx, ch):
    cals.set_data(ind=idx)
    d = cals.get_data()[idx].collection[ch]
    return np.asarray(d.t, float), np.array(d.x, dtype=float, copy=True)


def run_all(cals, voltages, files_per_voltage=3, channels=(0, 1), show=True):
    results = []
    for vi, V in enumerate(voltages):
        ncol = files_per_voltage * len(channels)
        if show:
            fig, ax = plt.subplots(2, ncol, figsize=(5 * ncol, 9), squeeze=False)
        k = 0
        for f in range(files_per_voltage):
            idx = files_per_voltage * vi + f
            for ch in channels:
                r = analyze(*load(cals, idx, ch))
                r.update(V=V, idx=idx, ch=ch)
                results.append(r)
                if show:
                    a = ax[0][k]
                    a.plot(r["td"], r["env"], color="C1", lw=0.8)
                    a.set_title(f"idx {idx} ch {ch}  {'OK' if r['usable'] else 'REJECT'}")
                    b = a.twinx()
                    b.plot(r["td"], r["s"], color="C0", lw=0.8, alpha=0.6)
                    a2 = ax[1][k]
                    step = max(1, len(r["u"]) // 4000)
                    a2.plot(r["u"][::step], r["q"][::step] / max(r["S"], 1e-15), ".", ms=1)
                    th = np.linspace(0, 2 * np.pi, 200)
                    a2.plot(np.cos(th), np.sin(th), "k", lw=0.8)
                    a2.set_aspect("equal")
                    a2.set_title(f"S/B={r['R']:.2e}  pp={r['stroke_pp']*1e9:.3f} nm\n"
                                 f"sweeps={r['sweeps']} SNR={r['snr']:.1f} ext={r['ext_ratio']:.2f}", fontsize=9)
                k += 1
        if show:
            ax[0][0].set_xlabel("Time, t (s)")
            ax[0][0].set_ylabel("109 kHz amplitude (V)  [blue: fringe signal]")
            ax[1][0].set_xlabel("cos(phi) from fringe signal")
            ax[1][0].set_ylabel("q / S  (signed 109 kHz)")
            fig.suptitle(f"Calibration scan, piezo driven at {V} Vpp")
            plt.tight_layout()
            plt.show()
    return results


def summarize(results, show=True):
    print(f"{'V':>5} {'idx':>4} {'ch':>3} {'B (V)':>9} {'S (V)':>10} {'S/B':>10} "
          f"{'pp (nm)':>9} {'sweeps':>6} {'SNR':>6} {'ext':>5}  use")
    for r in results:
        print(f"{r['V']:>5} {r['idx']:>4} {r['ch']:>3} {r['B']:>9.4f} {r['S']:>10.3e} "
              f"{r['R']:>10.3e} {r['stroke_pp']*1e9:>9.4f} {r['sweeps']:>6} "
              f"{r['snr']:>6.1f} {r['ext_ratio']:>5.2f}  {'yes' if r['usable'] else 'no'}")
    Vs = sorted({r["V"] for r in results})
    mean, sem = [], []
    for V in Vs:
        vals = np.array([r["stroke_pp"] for r in results if r["V"] == V and r["usable"]])
        mean.append(vals.mean() if vals.size else np.nan)
        sem.append(vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else np.nan)
    if show:
        plt.figure(figsize=(6, 4))
        for ch, mk in ((0, "o"), (1, "s")):
            pts = [(r["V"], r["stroke_pp"] * 1e9) for r in results if r["ch"] == ch and r["usable"]]
            if pts:
                plt.plot(*zip(*pts), mk, alpha=0.4, label=f"ch {ch}")
        plt.errorbar(Vs, np.array(mean) * 1e9, yerr=np.array(sem) * 1e9, fmt="k-", capsize=3,
                     label="mean ± SEM")
        plt.xlabel("Drive (Vpp)")
        plt.ylabel("Peak-to-peak stroke (nm)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    return Vs, mean, sem

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
from scipy.signal import periodogram, hilbert
sys.path.append("../")
from analysis import System

def wave(t, A, phi, offset):
    return np.abs(A) * np.sin(109e3 * 2 * np.pi * t - phi) + offset
def interference(x, A, offset):
    return np.abs(A) * np.power(np.cos(x * np.pi / 2 / 1064e-9), 2) + offset
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
def find_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

voltages = ["1", "2", "3", "4", "5", "5.5", "6", "6.5", "7", "7.5", "8", "8.5", "9", "9.5", "10"]
cals = System("michelson", ["../../data/20261006/2nd/piezo_char_" + voltages[v] + "V_109kHz_calibration/iter_" + str(i) + ".tdms" for v in range(len(voltages)) for i in range(0, 3)], set_data = True, channel = "X")

# Usage (fresh kernel, unfiltered data):
results = run_all(cals, voltages, show = False)
Vs, mean, sem = summarize(results)
print(mean)
print(sem)