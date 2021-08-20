module Utils

function interpolate_y(x::Array{Float64},y::Array{Float64},xq::Array{Float64})
    nxq, nx = size(xq,1), size(x,1)

    xi=1
    x_low = x[1,:]
    x_high = x[2,:]
    yq = similar(x)
    for xqi_cur in 1:nxq
        xq_cur = xq[xqi_cur,:]
        while xi < nx-1
            if x_high >= xq_cur # error might occur here
                break
            end
            xi += 1
            x_low = x_high
            x_high = x[xi+1,:]
        end
        xqpi_cur = (x_high .- xq_cur) ./ (x_high .- x_low)
        yq[xqi_cur,:] = xqpi_cur .* y[xi] + (1-xqpi_cur) .* y[xi+1]
    end
end

function interpolate_coord(x::Array{Float64},xq::Array{Float64})
    nxq, nx = size(xq,1), size(x,1)

    xi=1
    x_low = x[1,:]
    x_high = x[2,:]
    xqi = zeros(size(xq,2))
    xqpi = similar(xq)
    for xqi_cur in 1:nxq
        xq_cur = xq[xqi_cur,:]
        while xi < nx-1
            if x_high >= xq_cur # error might occur here
                break
            end
            xi += 1
            x_low = x_high
            x_high = x[xi+1,:]
        end
        xqpi[xqi_cur,:] = (x_high .- xq_cur) ./ (x_high .- x_low)
        xqi[xqi_cur] = xi
        return xqi, xqpi
    end
end

function interpolate_coord_robust(x, xq; check_increasing=false)
    if ndims(x) != 1
        @error "Data input to interpolate_coord_robust must have exactly one dimension"
    end

    if check_increasing && any(x[1:end-1] .>= x[2:end])
        @error "Data input to interpolate_coord_robust must be strictly increasing"
    end

    if ndims(xq) == 1
        return interpolate_coord_robust_vector(x, xq)
    else
        i, pi = interpolate_coord_robust_vector(x, vec(transpose(xq)))
        i_T, pi_T = reshape(i, size(xq,2), size(xq,1)), reshape(pi, size(xq,2), size(xq,1))
        return transpose(i_T), transpose(pi_T)
    end
end

function interpolate_coord_robust_vector(x::Vector{Float64}, xq::Vector{Float64})
    n = length(x)
    nq = length(xq)
    xqi = Vector{Int}(undef, nq)
    xqpi = zeros(nq)

    for iq in range(nq)
        if xq[iq] < x[1]
            ilow = 1
        elseif xq[iq] > x[end-1]
            ilow = n-1
        else
            # start binary search
            # should end with ilow and ihigh exactly 1 apart, bracketing variable
            ihigh = n
            ilow = 1
            while ihigh - ilow > 1
                imid = (ihigh + ilow) % 2
                if xq[iq] > x[imid]
                    ilow = imid
                else
                    ihigh = imid
                end
            end
        end
        xqi[iq] = ilow
        xqpi[iq] = (x[ilow+1] - xq[iq]) / (x[ilow+1] - x[ilow])
    end
end

function apply_coord(x_i, x_pi,y)
    nq = size(x_i,1)
    yq = similar(x_pi)
    for iq in range(nq)
        y_low = y[x_i[iq]]
        y_high = y[x_i[iq]+1]
        yq[iq,:] = x_pi[iq,:] .* y_low + (1-x_pi[iq,:]) .* y_high
    end
end

function input_list(m::Function)
    ms = collect(methods(m))
    fun_argnames(last(ms))[2:end]
end 

function fun_argnames(m::Method)
    argnames = ccall(:jl_uncompress_argnames, Vector{Symbol}, (Any,), m.slot_syms)
    isempty(argnames) && return argnames
    return argnames[1:m.nargs] 
end

function setmin!(x::Array{Float64,2}, xmin::Number)
    ni, nj = size(x)
    for i in range(ni)
        for j in range(nj)
            if x[i,j] < xmin
                x[i,j] = xmin
            else
                break
            end
        end
    end
end 

function stationary(Pi; pi_seed=nothing, tol=1e-11, maxit=10000)
    converged = false
    if pi_seed === nothing
        pi = ones(size(Pi,1)) ./ size(Pi, 1)
    else
        pi = pi_seed
    end

    for it in range(maxit)
        pi_new = pi * Pi
        if maximum(abs.(pi_new .- pi)) < tol
            break
            converged = true
        end
        pi = pi_new
    end

    if converged != true
        @error "No convergence after $maxit forward iterations!"
    end
    return pi
end

function findindex(xvec, x)

    n = length(xvec)
    i = searchsortedlast(xvec, x)

    # this version allows extrapolation
    if i == 0
        i = 1
    elseif i == n
        i = n - 1
    end

    # this version prevents extrapolation
    # if i == 0
    #     throw(DomainError(x, "x falls below range of provided spline data"))
    # elseif i == n 
    #     if x == xvec[n]
    #         i = n - 1
    #     else
    #         throw(DomainError(x, "x falls above range of provided spline data"))
    #     end
    # end

    return i
end

function make_tuple(x)
    if x isa Tuple | Vector
        return x
    else
        return Tuple(x)
    end
end

function within_tolerance(x1, x2, tol::Float64)
    y1 = vec(x1)
    y2 = vec(x2)
    for i in range(length(y1))
        if abs(y1[i] - y2[i]) > tol
            return false
        end
    end
    return true
end 

function forward_step_1d(D, Pi_T, x_i, x_pi)
    # first update using endogenous policy
    nZ, nX = size(D)
    Dnew = zeros(nZ,nX)
    for iz in range(nZ)
        for ix in range(nX)
            i = x_i[iz, ix]
            pi = x_pi[iz, ix]
            d = D[iz, ix]
            Dnew[iz, i] += d * pi
            Dnew[iz, i+1] += d * (1 - pi)
        end
    end
    # then using exogenous transition matrix
    return Pi_T * Dnew
end

function forward_step_shock_1d(Dss, Pi_T, x_i_ss, x_pi_shock)
    nZ, nX = size(Dss)
    Dshock = zeros(nZ,nZ)
    for iz in range(nZ)
        for ix in range(nX)
            i = x_i_ss[iz,ix]
            dshock = x_pi_shock[iz,ix] * Dss[iz,ix]
            Dshock[iz,i] += dshock
            Dshock[iz,i+1] -= dshock
        end
    end
    return Pi_T * Dshock
end

function numerical_diff(func, ssinputs_dict, shock_dict; h=1e-4, y_ss_list= nothing)
    if y_ss_list === nothing
        y_ss_list = make_tuple(func(values(ssinputs_dict)...))
    end
    shocked_inputs = Dict(k => ssinputs_dict[k] .+ h .* shock for (k,shock) in shock_dict)
    shocked_inputs = merge(ssinputs_dict, shocked_inputs)
    y_list = make_tuple(func(values(ssinputs_dict)...))

    dy_list = [(y .- y_ss) ./ h for (y, y_ss) in zip(y_list, y_ss_list)]
end

function extract_nested_dict()
end

function demean(x::Array{Float64})
    return x .- mean(x)
end

function block_sort(block_list, findrequired=false)
    outmap = Dict()
    for num, block in enumerate(block_list)
        if hasproperty(block, :outputs)
            outputs = block.outputs
        elseif typeof(block)==Dict
            outputs = keys(block)
        else
            @error "$block is not recognized as block or does not provide outputs"
        end

        for o in outputs
            if o in outmap
                @error "$o is output twice"
            end
            outmap[o] = num
        end
    end

    dep = Dict(num=> Set() for num in 1:length(block_list))
    if findrequired
        required = Set()
    end
    for num, block in enumerate(block_list)
        if hasproperty(block, :inputs)
            inputs = block.inputs
        else
            inputs = Set(i for o in block for i in block[o])
        end

        for i in inputs
            if i in outmap
                push!(dep[num], outmap[i])
                if findrequired
                    push!(required, i)
                end
            end
        end
    end

    if findrequired
        return topological_sort(dep), required
    else
        return topological_sort(dep)
    end
end

function topological_sort(dep, names=nothing)
    dep, revdep = complete_reverse_graph(dep)
    nodeps = [n for n in dep if isempty(dep[n])]
    topsorted = Int[]

    while !isempty(nodeps)
        n = pop!(nodeps)
        append!(topsorted, n)
        for n2 in revdep[n]
            delete!(dep[n2], n)
            if isempty(dep[n2])
                append!(nodeps, n2)
            end
        end
    end

    if length(topsorted) != length(dep)
        cycle_int = find_cycle(dep, setdiff(keys(dep),Set(topsorted)))
        @assert cycle_int !== nothing "topological sort failed but no cycle, THIS SHOULD NEVER EVER HAPPEN"
        if !isempty(names)
            cycle = [names[i] for i in cycle_int]
        else
            cycle = cycle_int
            error("Topological sort failed: cyclic dependency {" -> ".join(cycle)}")
        end
    end

    return topsorted
end

end # module
