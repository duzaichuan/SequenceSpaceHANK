using LinearAlgebra, OrderedCollections
include("utils.jl")

function curlyJ_sorted(block_list, inputs; ss=nothing, T=nothing, asymptotic=false, Tpost=nothing, save=false)
    topsorted, required = utils.block_sort(block_list, findrequired=true)

    curlyJs = Vector{Dict}()
    shocks = union(Set(inputs), required)
    for num in topsorted
        block = block_list[num]
        if typeof(block) == HetBlock
            if asymptotic
                J = ajac(block, ss, T=T, shock_list = [i for i in block.inputs if i in shocks], Tpost=Tpost, save = save, use_saved=use_saved)
            else
                J = jac(block, ss, T=T, shock_list = [i for i in block.inputs if i in shocks], Tpost=Tpost, save = save, use_saved=use_saved)
            end
        elseif typeof(block) == SimpleBlock
            J = jac(block, ss, shock_list = [i for i in block.inputs if i in shocks])
        else
            J = block
        end
        append!(curlyJs, J)
    end
    return curlyJs, required
end

function forward_accumulate(curlyJs, inputs; outputs=nothing, required=nothing)
    if outputs !== nothing && required !== nothing
        alloutputs = union(Set(outputs), Set(required))
    else
        alloutputs = nothing
    end

    jacflag = !isa(inputs, Dict)
    if jacflag
        out = Dict(i => Dict(i => IdentityMatrix()) for i in inputs)
    else
        out = copy(inputs)
    end

    for curlyJ in curlyJs
        if alloutputs !== nothing
            curlyJ = Dict(k => for (k,v) in curlyJ if k in alloutputs)
            if jacflag
                merge!(out, compose_jacobian(out, curlyJ))
            else
                merge!(out, apply_jacobian(out, curlyJ))
            end
        end
    end

    if outputs !== nothing
        return Dict(k => out[k] for k in outputs if k in keys(out))
    else
        if jacflag
            return Dict(k => v for (k,v) in out if k ∉ inputs)
        else
            return out
        end
    end
end

function compose_jacobian(jacdict2, jacdict1)
    jacdict = Dict()
    for (output, innerjac1) in jacdict1
        jacdict[output] = Dict()
        for (middle, jac1) in innerjac1
            innerjac2 = get(jacdict2, middle, Dict())
            for (inp, jac2) in innerjac2
                if inp ∈ keys(jacdict[output])
                    jacdict[output][inp] += jac1 @ jac2
                else
                    jacdict[output][inp] = jac1 @ jac2
                end
            end
        end
    end
end

function apply_jacobian(jacdict, indict)
    outdict = Dict()
    for (myout, innerjacdict) in jacdict
        for (myin, jac) in innerjacdict
            if myin ∈ indict
                if myout ∈ outdict
                    outdict[myout] += jac @ indict[myin]
                else
                    outdict[myout] = jac @ indict[myin]
                end
            end
        end
    end
end

function pack_jacobians(jacdict, inputs, outputs, T)

    nI, n0 = length(inputs), length(outputs)
    outjac = zeros(n0*T, nI*T)
     for i0 in range(n0)
         subdict = get(jacdict, outputs[i0], Dict())
     end
end
