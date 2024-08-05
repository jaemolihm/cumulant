function xlogabsx(x::Number)
    iszero(x) ? zero(x) : x * log(abs(x))
end

"""
    linear_spline_kramers_kronig(x, x1, x2, x3)
Kramers-Kronig transformation of a linear spline function at mesh points `x1`, `x2`, `x3`
evaluated at `x`.
Evaluate ``Pval ∫ dy b(y; x1, x2, x3) / (y - x)``, where ``b(y; x1, x2, x3)`` is a
piecewise-linear function with ``b(x1) = b(x3) = 0`` and ``b(x2) = 1``.
"""
function linear_spline_kramers_kronig(x, x1, x2, x3)
    (
        - xlogabsx(x - x1) / (x2 - x1)
        + xlogabsx(x - x2) * (x3 - x1) / (x2 - x1) / (x3 - x2)
        - xlogabsx(x - x3) / (x3 - x2)
    )
end

function apply_Kramers_Kronig(mesh, ys, x)
    N = length(mesh) - 2  # Excluding left and right endpoints
    mapreduce(+, 1:N) do i
        linear_spline_kramers_kronig(x, mesh[i], mesh[i+1], mesh[i+2]) * ys[i]
    end
end

"""
    run_cumulant_time_ordered(mesh, βs, ek, ts_coarse, ts_fine)
- `mesh`: mesh points of ``β(w)``
- `βs`: values of ``β(w) = -Im Σ(w + ek) / π`` at `mesh`
- `ek`: bare band energy (plus static correction)
- `ts_coarse`: small time mesh for cumulant calculation (where integral is evaluated explicitly)
- `ts_fine`: time mesh for cumulant interpolation (where integral is interpolated and padded)
"""
function run_cumulant_time_ordered(mesh, βs, ek, ts_coarse, ts_fine)
    βs_itp = linear_interpolation(mesh[2:end-1], βs; extrapolation_bc=0)

    # Step 1-1: Perform integration to get C(t) from Σ(w)
    Cs_t_small = tmap(ts_coarse) do t
        apply_cumulant_integral(mesh, βs, t)
    end


    # Step 1-2: Compute the constant and linear terms of C(t), subtract from C
    Σ0_real = -1 * apply_Kramers_Kronig(mesh, βs, 0.0)
    Σ0_imag = -π * βs_itp(0.0)
    Σ0 = Σ0_real + im * Σ0_imag

    δw = 1e-5
    dΣ0_real = -1 * (apply_Kramers_Kronig(mesh, βs, δw) - apply_Kramers_Kronig(mesh, βs, -δw)) / 2δw
    dΣ0_imag = -π * (βs_itp(δw) - βs_itp(-δw)) / 2δw
    dΣ0 = dΣ0_real + im * dΣ0_imag

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
