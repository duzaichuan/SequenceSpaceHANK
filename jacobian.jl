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

