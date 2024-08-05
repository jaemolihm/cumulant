using PolyLog

# Frohlich model for the electron-phonon interaction
# We use the atomic Hartree units (ħ = mₑ = e² = 4πε0 = 1).
#
# References
# [1] N. Kandolf et al., PRB 105, 085148 (2022)

struct FrohlichModel
    α  :: Float64
    w₀ :: Float64
    m  :: Float64
    μ  :: Float64
    T  :: Float64
end


Base.Broadcast.broadcastable(model::FrohlichModel) = Ref(model)

get_ek(k, model::FrohlichModel) = norm(k)^2 / 2 / model.m

function get_eph_g(q, model :: FrohlichModel)
    (; α, w₀, m) = model
    if norm(q) == 0
        return 0.0
    else
        return sqrt(4π * α * sqrt(w₀^3 / 2m)) / norm(q)
    end
end

function L(z1, z2)
    # Eq.(41) of Ref.[1]. (Typos on the sign of second and third terms fixed)
    li2((1 + z1) / (1 + z2)) + li2((1 + z1) / (1 - z2)) - li2((1 - z1) / (1 + z2)) - li2((1 - z1) / (1 - z2))
end



function occ_boson(e, T)
    if T > sqrt(eps(eltype(T)))
        return e == 0 ? zero(e) : 1 / expm1(e / T)
    elseif T >= 0
        return zero(e)
    else
        throw(ArgumentError("Temperature must be positive"))
    end
end

"""
    get_Σ_analytic(k, w, model::FrohlichModel)

The retarded self-energy of the undoped Frohlich model (equals the greater self-energy),
computed using the analytic formula.
For μ < 0, use Eq.(28) of Ref.[1]. (The π in the denominator is a typo and is removed.)
For μ > 0, use Eq.(39-42) of Ref.[1]. (The π in the denominator is a typo and is removed.)
"""
function get_Σ_analytic(k, w, model::FrohlichModel)
    (; w₀, α, μ, T) = model
    ek = get_ek(k, model)

    nq = occ_boson(w₀, T)

    if μ < 0
        if ek < eps(typeof(ek))
            # Case k = 0
            Σ_emi = -im * α * w₀^1.5 / √(w - w₀)
            Σ_abs = -im * α * w₀^1.5 / √(w + w₀)
        else
            # Case k /= 0
            Σ_emi = -im * α * w₀^1.5 / (2 * √(ek)) * log((√(w - w₀) + √(ek)) / (√(w - w₀) - √(ek)))
            Σ_abs = -im * α * w₀^1.5 / (2 * √(ek)) * log((√(w + w₀) + √(ek)) / (√(w + w₀) - √(ek)))
        end

        return Σ_emi * (nq + 1) + Σ_abs * nq

    else
        T > 0 && throw(ArgumentError("μ > 0 and T > 0 not implemented"))
        if ek < eps(typeof(ek))
            # Eq.(B9) of Ref.[1]
            Σles = log((√(conj(w) + w₀) + √(μ)) / (√(conj(w) + w₀) - √(μ))) / √(conj(w) + w₀)

            # Eq.(B11) of Ref.[1]
            Σgtr = -(log((√(w - w₀) + √(μ)) / (√(w - w₀) - √(μ))) + im * π) / √(w - w₀)

            return (conj(Σles) + Σgtr) * α * w₀^1.5 / π
        else
            # Eq.(39) of Ref.[1] without the last Σ(E_F) term
            Σles = -(
                L(√(μ / ek), √((conj(w) + w₀) / ek))
                + log((conj(w) + w₀ - μ) / (conj(w) + w₀ - ek)) * log(abs((√(μ) + √(ek)) / (√(μ) - √(ek))))
            )

            # Eq.(42) of Ref.[1] without the last Σ(E_F) term
            # (Typo on the sign of the denominator in the second term fixed)
            Σgtr = (
                L(√(μ / ek), √((w - w₀) / ek))
                + log((w - w₀ - μ) / (w - w₀ - ek)) * log(abs((√(μ) + √(ek)) / (√(μ) - √(ek))))
                - im * π * log((√(w - w₀) + √(ek)) / (√(w - w₀) - √(ek)))
            )

            return (conj(Σles) + Σgtr) * α * w₀^1.5 / 2π / √(ek)
        end

    end
end
