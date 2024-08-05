using PyPlot

# Load all the functions from files
include("cumulant_retarded.jl")
include("cumulant_time_ordered.jl")
include("frohlich.jl")


begin
    # ------------------------------------------------------------------------
    # INPUT PARAMETERS

    μ = 0.4  # Chemical potential
    T = 0.0  # Temperature
    kF = sqrt(2 * μ)  # Fermi wavevector (assuming m = 1.0: see get_bare_band)

    # Define time mesh for cumulant calculation.
    # dt determines maximum frequency box size by wmax = π / dt
    # tmax deterimes the frequency resolution by dw = π / tmax
    # For each t on ts_coarse, we calculate the cumulant integral explicitly.
    # We increment the resolution and box size by interpolating and extrapolating the
    # results to a finer mesh ts_fine.
    # For the extrapolation, we use the analytic linear form of the cumulant integral.
    tmax = 100.
    dt = 0.5

    # Wavevector range to compute the spectral function
    ks = range(0., 3., length=61)

    # Frequency mesh for the self-energy. The mesh may be nonuniform.
    mesh_Σ = vcat(
        range(-20., -3., step = 0.05),
        range(-3.,   3., step = 0.01)[2:end],
        range( 3.,  20., step = 0.05)[2:end],
    )

    function get_bare_band(k)
        # Returns bare band energy (plus static correction) at wavevector `k`
        m = 1.0
        return k^2 / (2m)
    end
    
    function get_self_energy(k, w)
        # Returns retarded self-energy at wavevector `k` and energy `w`
        # Momentum independent Lorentzian self-energy at w0 and width gamma
        w0 = 1.0
        gamma = 0.1
        return 0.5 / (w + w0 + im * gamma)
    end
    # ------------------------------------------------------------------------


    # Setup the time and frequency mesh for the cumulants
    ts_coarse = range(0, tmax, step=dt)
    ts_fine = range(0, tmax * 200, step = dt / 2)
    ws = fftfreq_t2w(ts_fine)
    inds_plot = searchsortedfirst(ws, -4.):searchsortedfirst(ws, 4.)


    # Dyson self-energy
    As_Dyson = tmapreduce(hcat, ks) do k
        ek = get_bare_band(k)
        Σs = @. get_self_energy.(k, ws)
        @. -imag(1 / (ws - ek - Σs)) / π
    end


    # Retarded cumulant self-energy
    @time As_cum_R = tmapreduce(hcat, ks) do k
        ek = get_bare_band(k)
        Σs = get_self_energy.(k, mesh_Σ[2:end-1] .+ ek)
        run_cumulant_retarded(mesh_Σ, Σs, ek, ts_coarse, ts_fine)
    end
    @info extrema(As_cum_R)
    As_cum_R[As_cum_R .< 0] .= 0


    # Time-ordered cumulant self-energy
    @time As_cum_T = tmapreduce(hcat, ks) do k
        ek = get_bare_band(k)
        Σs = @. get_self_energy.(k, mesh_Σ)
        βs = .-imag.(linear_interpolation(mesh_Σ, Σs; extrapolation_bc=0).(mesh_Σ[2:end-1] .+ ek)) ./ π

        if ek > μ
            βs[mesh_Σ[2:end-1] .+ ek .<= μ] .= 0
        else
            βs[mesh_Σ[2:end-1] .+ ek .>= μ] .= 0
        end
        run_cumulant_time_ordered(mesh_Σ, βs, ek, ts_coarse, ts_fine)
    end
    @info extrema(As_cum_T)
    As_cum_T[As_cum_T .< 0] .= 0

    # Time ordered cumulant for ek > μ is not implemented correctly.
    # (We need to implement Fourier transformation with Θ(t < 0) instead of Θ(t > 0).)
    As_cum_T[:, get_bare_band.(ks) .> μ] .= 1e-10

    # ------------------------------------------------------------------------
    # Plot results

    vmax = 5.0
    fig, plotaxes = subplots(1, 3, figsize=(12, 3); sharex=true, sharey=true)

    ws_ = ws[inds_plot]
    kwargs_plot = (; aspect="auto", origin="lower", extent=(ks[1], ks[end], ws_[1], ws_[end]),
        # vmin=0, vmax
        norm = matplotlib.colors.LogNorm(; vmin=vmax/1e3, vmax),
    )
    # plotaxes[1].imshow(As_cum_exact; kwargs_plot...)
    img = plotaxes[1].imshow(As_Dyson[inds_plot, :]; kwargs_plot...)
    colorbar(img; ax=plotaxes[1])

    img = plotaxes[2].imshow(As_cum_R[inds_plot, :]; kwargs_plot...)
    colorbar(img; ax=plotaxes[2])

    img = plotaxes[3].imshow(As_cum_T[inds_plot, :]; kwargs_plot...)
    colorbar(img; ax=plotaxes[3])
    plotaxes[1].set_title("Dyson")
    plotaxes[2].set_title("Retarded cumulant")
    plotaxes[3].set_title("Time-ordered cumulant")
    for ax in plotaxes
        ax.plot(ks, get_bare_band.(ks), "C1-", lw=1)
        ax.axhline(μ, c="r", lw=1, ls="--")
        ax.set_ylim([-4, 5])
        ax.set_ylim([-3, 4])
        ax.axvline(kF, c="r", ls="--", lw=1)
    end
    fig.savefig("fig_lorentzian.png"; bbox_inches="tight")
    display(fig); close(fig)
end



begin
    # ------------------------------------------------------------------------
    # INPUT PARAMETERS : Frohlich model

    # Define Frohlich model
    α = 0.5  # e-ph coupling strength
    w₀ = 1.0  # LO phonon frequency
    m = 1.0  # Mass
    μ = 0.4  # Chemical potential
    T = 0.0  # Temperature
    model = FrohlichModel(α, w₀, m, μ, T)

    kF = sqrt(2m * μ)  # Fermi wavevector

    # Define time mesh for cumulant calculation.
    # dt determines maximum frequency box size by wmax = π / dt
    # tmax deterimes the frequency resolution by dw = π / tmax
    # For each t on ts_coarse, we calculate the cumulant integral explicitly.
    # We increment the resolution and box size by interpolating and extrapolating the
    # results to a finer mesh ts_fine.
    # For the extrapolation, we use the analytic linear form of the cumulant integral.
    tmax = 100.
    dt = 0.5

    # Wavevector range to compute the spectral function
    ks = range(0., 3., length=61)

    # Frequency mesh for the self-energy. The mesh may be nonuniform.
    mesh_Σ = vcat(
        range(-20., -3., step = 0.05),
        range(-3.,   3., step = 0.01)[2:end],
        range( 3.,  20., step = 0.05)[2:end],
    )

    function get_bare_band(k)
        # Returns bare band energy (plus static correction) at wavevector `k`
        get_ek(k, model)
    end
    
    function get_self_energy(k, w)
        # Returns retarded self-energy at wavevector `k` and energy `w`
        # Momentum independent Lorentzian self-energy at w0 and width gamma
        gamma = 0.1
        get_Σ_analytic(k, w + im * gamma, model)
    end

    # ------------------------------------------------------------------------


    # Setup the time and frequency mesh for the cumulants
    ts_coarse = range(0, tmax, step=dt)
    ts_fine = range(0, tmax * 200, step = dt / 2)
    ws = fftfreq_t2w(ts_fine)
    inds_plot = searchsortedfirst(ws, -4.):searchsortedfirst(ws, 4.)


    # Dyson self-energy
    As_Dyson = tmapreduce(hcat, ks) do k
        ek = get_bare_band(k)
        Σs = @. get_self_energy.(k, ws)
        @. -imag(1 / (ws - ek - Σs)) / π
    end


    # Retarded cumulant self-energy
    @time As_cum_R = tmapreduce(hcat, ks) do k
        ek = get_bare_band(k)
        Σs = get_self_energy.(k, mesh_Σ[2:end-1] .+ ek)
        run_cumulant_retarded(mesh_Σ, Σs, ek, ts_coarse, ts_fine)
    end
    @info extrema(As_cum_R)
    As_cum_R[As_cum_R .< 0] .= 0


    # Time-ordered cumulant self-energy
    @time As_cum_T = tmapreduce(hcat, ks) do k
        ek = get_bare_band(k)
        Σs = @. get_self_energy.(k, mesh_Σ)
        βs = .-imag.(linear_interpolation(mesh_Σ, Σs; extrapolation_bc=0).(mesh_Σ[2:end-1] .+ ek)) ./ π

        if ek > μ
            βs[mesh_Σ[2:end-1] .+ ek .<= μ] .= 0
        else
            βs[mesh_Σ[2:end-1] .+ ek .>= μ] .= 0
        end
        run_cumulant_time_ordered(mesh_Σ, βs, ek, ts_coarse, ts_fine)
    end
    @info extrema(As_cum_T)
    As_cum_T[As_cum_T .< 0] .= 0

    # Time ordered cumulant for ek > μ is not implemented correctly.
    # (We need to implement Fourier transformation with Θ(t < 0) instead of Θ(t > 0).)
    As_cum_T[:, get_bare_band.(ks) .> μ] .= 1e-10


    # ------------------------------------------------------------------------
    # Plot results

    vmax = 5.0
    fig, plotaxes = subplots(1, 3, figsize=(12, 3); sharex=true, sharey=true)

    ws_ = ws[inds_plot]
    kwargs_plot = (; aspect="auto", origin="lower", extent=(ks[1], ks[end], ws_[1], ws_[end]),
        # vmin=0, vmax
        norm = matplotlib.colors.LogNorm(; vmin=vmax/1e3, vmax),
    )
    # plotaxes[1].imshow(As_cum_exact; kwargs_plot...)
    img = plotaxes[1].imshow(As_Dyson[inds_plot, :]; kwargs_plot...)
    colorbar(img; ax=plotaxes[1])

    img = plotaxes[2].imshow(As_cum_R[inds_plot, :]; kwargs_plot...)
    colorbar(img; ax=plotaxes[2])

    img = plotaxes[3].imshow(As_cum_T[inds_plot, :]; kwargs_plot...)
    colorbar(img; ax=plotaxes[3])
    plotaxes[1].set_title("Dyson")
    plotaxes[2].set_title("Retarded cumulant")
    plotaxes[3].set_title("Time-ordered cumulant")
    for ax in plotaxes
        ax.plot(ks, get_bare_band.(ks), "C1-", lw=1)
        ax.axhline(μ, c="r", lw=1, ls="--")
        ax.set_ylim([-4, 5])
        ax.set_ylim([-3, 4])
        ax.axvline(kF, c="r", ls="--", lw=1)
    end
    fig.savefig("fig_frohlich.png"; bbox_inches="tight")
    display(fig); close(fig)
end
