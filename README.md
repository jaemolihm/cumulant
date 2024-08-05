# Cumulant

### Installing the dependencies
```bash
julia -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
```

### Running the script
```bash
julia -t 8 --project=. main.jl
```

Modify `-t 8` to use different number of threads than 8.


### Model Lorentzian self-energy
![Lorentzian](fig_lorentzian.png)

### Frohlich model
![Frohlich](fig_frohlich.png)
