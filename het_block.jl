using Parameters, LinearAlgebra, Dictionaries
include("/home/zaichuan/Projects/sequence-space-solving-ha-models/utils.jl")

@with_kw mutable struct HetBlock @deftype Vector{Symbol}
    exogenous::Symbol=:Pi
    policy
    backward
    all_outputs # variables after the `return` of the backward step function
    all_inputs
    inputs_p = union(exogenous,backward)
    non_back_outputs = setdiff(all_outputs,backward)
    # aggregate outputs and inputs for utils.block_sort
    inputs = setdiff(replace(all_inputs, Symbol(exogenous,"_p")=>exogenous), [Symbol(k, "_p") for k in backward])
    outputs = [Symbol(uppercase(String(k))) for k in non_back_outputs]
    hetinput = nothing
    hetinput_inputs
    hetinput_outputs_order
    saved::Dict{Symbol,Array}
    prelim_saved::Dict{Symbol,Array}
    saved_shock_list
    saved_output_list
end


function ss(m::HetBlock, back_step_fun; backward_tol=1E-8, backward_maxit=5000, forward_tol=1E-10, forward_maxit=100_000, kwargs...)
    # extract information from kwargs
    kwargs = Dict(kwargs)
    Pi = kwargs[m.exogenous]
    grid = Dict(k => kwargs[Symbol(k,"_grid")] for k in m.policy)
    D_seed = get(kwargs,:D, nothing)
    pi_seed = get(kwargs, Symbol(m.exogenous,"_seed"), nothing)

    # run backward iteration
    sspol = policy_ss(m, back_step_fun, backward_tol, backward_maxit, kwargs...)
    # run forward iteration
    D = dist_ss(Pi, sspol, grid, forward_tol, forward_maxit, D_seed, pi_seed)
    # aggregate all outputs other than backward variables on grid, capitalize
    aggs = Dict(Symbol(uppercase(String(k))) => dot(D, sspol[k]) for k in m.non_back_outputs)

    ss_dict = merge(sspol, aggs)
    ss_dict[:D] = D
end

function policy_ss(m::HetBlock, back_step_fun::Function;tol=1e-8,maxit=5000, ssin...)
    original_ssin = ssin # Dict(ssin) ?
    ssin = make_inputs(m,ssin)
    old = Dict() # No content type specified may be inefficient
    sspol = Dict()
    converged= false
    for it in range(maxit)
        try
            sspol = Dict(k => v for (k, v) in zip(m.all_outputs, back_step_fun(values(ssin)...))) # back_step_fun cannot directly read keywords in ssin..., so the order of ssin dictionary is important (dictionary vs. Dict)
        catch e
            @warn "Missing input $e"
        end

        if mod(it,10) ==1 && all(utils.within_tolerance(sspol[k], old[k], tol) for k in m.policy)
            break
            converged = true
        end
        merge!(old, Dict(k => sspol[k] for k in m.policy))
        merge!(ssin, Dict(Symbol(k,"_p") => sspol[k] for k in m.backward))
    end
    if converged != true
        @error "No convergence of policy functions after $maxit backward iterations!"
    end

    for k in m.inputs_p
        ssin[k] = ssin[Symbol(k,"_p")]
        delete!(ssin, Symbol(k,"_p"))
    end
    if isnothing(m.hetinput) != true
        for k in m.hetinput_inputs
            if haskey(original_ssin, k)
                ssin[k] = original_ssin[k]
            end
        end
    end
    return merge(ssin, sspol)
end

function dist_ss(m::HetBlock, Pi, sspol,grid; tol=1e-10, maxit=100000, D_seed= nothing, pi_seed= nothing)
    converged = false
    if D_seed === nothing
        pi = utils.stationary(Pi, pi_seed)
        endogenous_dims = [size(grid[k], 1) for k in m.policy]
        mesh_dims = copy(endogenous_dims)
        pushfirst!(mesh_dims,1) # pi in julia is a vector
        D = repeat(pi, outer=mesh_dims) ./ prod(endogenous_dims)
    else
        D = D_seed
    end
    sspol_i = Dict()
    sspol_pi = Dict()

    for pol in m.policy
        sspol_i[pol], sspol_pi[pol] = utils.interpolate_coord_robust(grid[pol], sspol[pol])
    end
    Pi_T = copy(transpose(Pi))

    for it in range(maxit)
        Dnew = forward_step(D, Pi_T, sspol_i, sspol_pi)

        if mod(it, 10) == 0 && utils.within_tolerance(D,Dnew, tol)
            break
            converged = true
        end
        D = Dnew
    end
    if converged != true
        @error "No convergence after $maxit forward iterations"
    end
    return D
end

function jac(m::HetBlock, back_step_fun, ss, T, shock_list; output_list = nothing, h= 1e-4, save=false, use_saved=false)
    if output_list === nothing
        output_list = m.non_back_outputs
    end
    if use_saved
        return utils.extract_nested_dict(savedA=m.saved[:J], keys1=[Symbol(uppercase(String(o))) for o in output_list], shape=(T,T))
    end
    ssin_dict, Pi, ssout_list, ss_for_hetinput, sspol_i, sspol_pi, sspol_space = jac_prelim(m, back_step_fun, ss, save= save)

    curlyYs, curlyDs = Dict(), Dict()
    for i in shock_list
        curlyYs[i], curlyDs[i] = backward_iteration_fakenews(i, output_list, ssin_dict, ssout_list,
                                                             ss[:D], copy(transpose(Pi)), sspol_i,
                                                             sspol_pi, sspol_space, T, h, ss_for_hetinput)
    end
    curlyPs = Dict()
    for o in output_list
        curlyPs[o] = forward_iteration_fakenews(ss[o], Pi, sspol_i, sspol_pi, T-1)
    end

    F = Dict(uppercase(o) => Dict() for o in output_list)
    J = Dict(uppercase(o) => Dict() for o in output_list)
    for o in output_list
        for i in shock_list
            F[uppercase(o)][i]= build_F(curlyYs[i][o], curlyDs[i], curlyPs[o])
            J[uppercase(o)][i] = J_from_F(F[uppercase(o)][i])
        end
    end

    if save
        m.saved_shock_list, m.saved_output_list = shock_list, output_list
        m.saved = Dict(:curlyYs => curlyYs, :curlyDs => curlyDs, :curlyPs => curlyPs, :F => F, :J => J)
    end
    return J
end

function jac_prelim(m::HetBlock,back_step_fun, ss; save= false, use_saved=false)
    output_names = [:ssin_dict, :Pi, :ssout_list, :ss_for_hetinput, :sspol_i, :sspol_pi, :sspol_space]
    if use_saved
        if m.prelim_saved
            return Tuple(m.prelim_saved[k] for k in output_names)
        else
            @error "Nothing saved to be used by jac_prelim"
        end
    end

    ssin_dict = make_inputs(m, ss)
    Pi = ss[m.exogenous]
    grid = Dict(k => ss[Symbol(k, "_grid")] for k in m.policy)
    ssout_list = back_step_fun(values(ssin_dict)...)

    ss_for_hetinput = nothing
    if m.hetinput !== nothing
        ss_for_hetinput = Dict(k => ss[k] for k in m.hetinput_inputs if k in ss)
    end

    sspol_i = Dict()
    sspol_pi = Dict()
    sspol_space = Dict()
    for pol in m.policy
        sspol_i[pol], sspol_pi[pol] = utils.interpolate_coord_robust(grid[pol], ss[pol])
        sspol_space[pol]= grid[pol][sspol_i[pol]+1] .- grid[pol][sspol_i[pol]] # sspol_i[pol] should be a scalar?
    end

    toreturn = (ssin_dict, Pi, ssout_list, ss_for_hetinput, sspol_i, sspol_pi, sspol_space)
    if save
        m.prelim_saved = Dict(k => v for (k,v) in zip(output_names, toreturn))
    end
    return toreturn
end

function make_inputs(m::HetBlock, indict)
    if isnothing(m.hetinput) != true
        outputs_as_tuple = utils.make_tuple(m.hetinput) # argument form in make_tuple is wierd        
    end
    indict_new = Dict(k => indict[k] for k in setdiff(m.all_inputs, m.inputs_p) if k in indict)
    try
        return merge(indict_new, Dict(Symbol(k,"_p") => indict[k] for k in m.inputs_p))
    catch e
        print("Missing backward variable or Markov matrix $e !") 
    end
end

function backward_iteration_fakenews(m::HetBlock, input_shocked, output_list, ssin_dict, ssout_list,  Dss, Pi_T, sspol_i, sspol_pi, sspol_space, T; h=1e-4, ss_for_hetinput=nothing)
    
    if m.hetinput !== nothing && input_shocked in m.hetinput_inputs
        din_dict = Dict(zip(m.hetinput_outputs_order, utils.numerical_diff_symmetric(m.hetinput, ss_for_hetinput, Dict(input_shocked => 1) , h)))
    else
        din_dict = Dict(input_shocked => 1)
    end

    curlyV, curlyD, curlyY = backward_step_fakenews(din_dict, output_list, ssin_dict, ssout_list, Dss, Pi_T, sspol_i, sspol_pi, sspol_space, h)
    
    curlyDs = zeros(T,size(curlyD,1),size(curlyD,2))
    curlyYs = Dict(k => zeros(T) for k in keys(curlyY))

    curlyDs[1,:,:] = curlyD
    for k in keys(curlyY)
        curlyYs[k][1] = curlyY[k]
    end

    for t in 2:T
        curlyV, curlyDs[t,:,:], curlyY = backward_step_fakenews(Dict(Symbol(k,"_p") => v for (k,v) in curlyV),
                                                                output_list, ssin_dict, ssout_list, Dss, Pi_T, sspol_i, sspol_pi, sspol_space, h)
        for k in curlyY
            curlyYs[k][t] = curlyY[k]
        end
    end
    return curlyYs, curlyDs
end

function backward_step_fakenews(m::HetBlock, back_step_fun, din_dict, output_list, ssin_dict, ssout_list, Dss, Pi_T, sspol_i, sspol_pi, sspol_space; h=1E-4)
    shocked_outputs = Dict(k => v for (k,v) in zip(m.all_outputs,
                                                   utils.numerical_diff(back_step_fun, ssin_dict, din_dict, h, ssout_list)))

    curlyV = Dict(k => shocked_outputs[k] for k in m.backward)
    pol_pi_shock = Dict(k => -shocked_outputs[k]./sspol_space[k] for k in m.policy) # sspol_space is the denominator in "_pi" variables? Values in pol_pi_shock are analogous to xqpi, while values in -shocked_outputs are analogous to (x[ilow+1] - xq[iq]) in function interpolate_coord_robust_vector.
    curlyD = forward_step_shock(m::HetBlock, Dss, Pi_T, sspol_i, sspol_pi, pol_pi_shock)
    
    curlyY = Dict(k => dot(Dss, shocked_outputs[k]) for k in output_list)
    return curlyV, curlyD, curlyY
end

function forward_step(m::HetBlock, D, Pi_T, pol_i, pol_pi)
    if length(m.policy) == 1
        p, = m.policy
        return utils.forward_step_1d(D, Pi_T, pol_i[p], pol_pi[p])
    elseif  length(m.policy) == 2
        p1, p2 = m.policy
        return utils.forward_step_2d(D, Pi_T, pol_i[p1], pol_i[p2], pol_pi[p1], pol_pi[p2])
    else
        @error "length of policy variables only up to 2 implemented!"
    end
end

function forward_step_shock(m::HetBlock, Dss, Pi_T, pol_i_ss, pol_pi_ss, pol_pi_shock)
    if length(m.policy) == 1
        p, =m.policy
        return utils.forward_step_shock_1d(Dss, Pi_T, pol_i_ss[p], pol_pi_shock[p])
    elseif  length(m.policy) ==2
        p1,p2 = m.policy
        return utils.forward_step_shock_2d(Dss, Pi_T, pol_i_ss[p1], pol_i_ss[p2],
                                           pol_pi_ss[p1], pol_pi_ss[p2], pol_pi_shock[p1], pol_pi_shock[p2])
    else
        @error "$length(m.policy) policy variables, only up to 2 implemented!"
    end
end
