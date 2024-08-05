using LinearAlgebra
using StaticArrays
using Interpolations
using FFTW
using SpecialFunctions
using OhMyThreads


function _cumulant_int0(x, t)
    # ∫ dw (exp(-iwt) - 1 + iwt) / w^2
    @assert t >= 0
    if x > 0
        -im * t * (im * (-π/2 - sinint(t * x) ) + cosint(t * x)) - cis(-t * x) / x + im * t * log(x) + 1 / x
    else
        -im * t * (im * (-π/2 + sinint(-t * x) ) + cosint(-t * x)) - cis(-t * x) / x + im * t * log(-x) + 1 / x
    end
end
function _cumulant_int1(x, t)
    # ∫ dw w * (exp(-iwt) - 1 + iwt) / w^2
    @assert t >= 0
    if x > 0
        im * (-π/2 - sinint(t * x) ) + cosint(t * x)  + im * t * x - log(x)
    else
        im * (-π/2 + sinint(-t * x) ) + cosint(-t * x)  + im * t * x - log(-x)
    end
end


# Evaluate ∫ b(w; a, b, c) * (exp(-iwt) - 1 + iwt) / w^2 for a linear spline function b
function _cumulant_integral(t, a, b, c)
    if t == 0.
        return 0.0im
    end
    a == 0 && (a += 1e-5)
    b == 0 && (b += 1e-5)
    c == 0 && (c += 1e-5)

    int0_a = _cumulant_int0(a, t)
    int0_b = _cumulant_int0(b, t)
    int0_c = _cumulant_int0(c, t)
    int1_a = _cumulant_int1(a, t)
    int1_b = _cumulant_int1(b, t)
    int1_c = _cumulant_int1(c, t)

    z  = -a / (b - a) * (int0_b - int0_a)
    z +=  1 / (b - a) * (int1_b - int1_a)
    z +=  c / (c - b) * (int0_c - int0_b)
    z += -1 / (c - b) * (int1_c - int1_b)
    z
end


function apply_cumulant_integral(mesh, βs, t)
    N = length(mesh) - 2  # Excluding left and right endpoints
    tmapreduce(+, 1:N) do i
        a, b, c = mesh[i], mesh[i+1], mesh[i+2]
        _cumulant_integral(t, a, b, c) * βs[i]
    end
end

function fftfreq_t2w(ts_fine)
    # Frequency mesh for the Fourier transformation of y(t) to y(w).
    # ts_fine includes only 0 and positive t values.
    @assert ts_fine[1] == 0.0
    dt = ts_fine[2] - ts_fine[1]
    wmax = π / dt
    ws = range(-wmax, wmax, length = 2length(ts_fine)+1)[1:end-1]
    ws
end


function fft_t2w(ts_fine, ys_t)
    # Fourier transform of y(t) to y(w).
    # Assume ys(t) is nonzero only for t >= 0 and only t >= 0 data are used.
    @assert ts_fine[1] == 0.0
    @assert length(ts_fine) == length(ys_t)
    dt = ts_fine[2] - ts_fine[1]
    wmax = π / dt
    ws = range(-wmax, wmax, length = 2length(ts_fine)+1)[1:end-1]

    # Pad t < 0 data with zeros.
    ys_t_pad = vcat(zeros(length(ts_fine)), ys_t)
    ys = fftshift(bfft(fftshift(ys_t_pad))) .* dt
    (; ys, ws)
end

"""
    run_cumulant(mesh, Σs, ek, ts_coarse, ts_fine)
- `mesh`: mesh points of self-energy
- `Σs`: values of ``Σ(w + ek)`` at `mesh`
- `ek`: bare band energy (plus static correction)
- `ts_coarse`: small time mesh for cumulant calculation (where integral is evaluated explicitly)
- `ts_fine`: time mesh for cumulant interpolation (where integral is interpolated and padded)
"""
function run_cumulant_retarded(mesh, Σs, ek, ts_coarse, ts_fine)
    Σs_itp = linear_interpolation(mesh[2:end-1], Σs; extrapolation_bc=0)
    
    # Step 1-1: Perform integration to get C(t) from Σ(w)
    βs = -imag.(Σs) ./ π
    Cs_t_small = tmap(ts_coarse) do t
        apply_cumulant_integral(mesh, βs, t)
    end

    # Step 1-2: Compute the constant and linear terms of C(t), subtract from C
    δw = 1e-5
    Σ0 = Σs_itp(0.0)
    dΣ0 = (Σs_itp(δw) - Σs_itp(-δw)) / 2δw

    @. Cs_t_small -= -im * Σ0 * ts_coarse + dΣ0

    # Step 1-3: Interpolation C(t) and append linear extrapolation
    Cs_t_itp = linear_interpolation(ts_coarse, Cs_t_small; extrapolation_bc=0)
    Cs_t = @. Cs_t_itp(ts_fine) - im * Σ0 * ts_fine + dΣ0


    # Step 2: Perform Fourier transformation to get A_cum(t) from C(t)
    Gs_cum_t = -im .* cis.(-ek .* ts_fine) .* exp.(Cs_t)
    Gs_cum_t[1] /= 2  # Divide t=0 term by 2
    Gs_cum, ws = fft_t2w(ts_fine, Gs_cum_t)

    As_cum = @. -imag(Gs_cum) / π

    As_cum
end
