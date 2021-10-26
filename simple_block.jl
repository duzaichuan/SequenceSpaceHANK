module Simple

using MacroTools, Parameters, LinearAlgebra, OrderedCollections
include("utils.jl")

@with_kw struct SimpleBlock
    # f # static function
    dyn_eqs # dynamic equations in dynare form
    input_list::Vector
    output_list
    time_disp::Set # record non-zero time displacements i in dynare-like eqs
end

macro simple(f_expr)
    @capture(f_expr, function fname_(inputs__) body__ end)
    # @static_function(f_expr)
    @show body
    body_homo = [homogeneize(body[i]) for i in 1:length(body)-1]
    body_static = map(x -> to_static(x, inputs), body_homo)
    new_body = [Expr(:(=), body_static[i].args[2:end]...) for i in 1:length(body_static)]
    return esc(quote
                   $fname = SimpleBlock(dyn_eqs = $(body[1:end-1]),
                                        input_list = $inputs,
                                        output_list = $(body[end].args),
                                        time_disp = Set())
                   function f($(inputs...))
                       $(new_body...)
                       $(body[end])
                   end
               end)
end

function jac(m::SimpleBlock, ss, f; T=nothing, shock_list=nothing, h=1e-5)

    if shock_list === nothing
        shock_list = m.input_list
    end
    raw_derivatives = Dict(o => Dict() for o in m.output_list)

    n_endo = length(shock_list)
    d = Dict{Symbol,Tuple{Int,Int}}()
    for i = 1:n_endo
        d[shock_list[i]] = (0,0)
    end
    for j = 1:n_endo
        scan_expr(m.dyn_eqs[j], shock_list, d)
    end
    x_ss = OrderedDict{Symbol,Float64}()

    for ii in m.input_list
        x_ss[ii] = ss[ii]
    end

    for k in shock_list
        x_ss_up[k] = ss[k] + h
        x_ss_lo[k] = ss[k] - h

        arglist_up = values(NamedTuple(x_ss_up))
        arglist_lo = values(NamedTuple(x_ss_lo))

        y_up_all = utils.make_tuple(f(arglist_up...))
        y_down_all = utils.make_tuple(f(arglist_lo...))

        (la,le) = lag_lead[k]
        if la == -1 && le ==0
            for y_up, y_down, o in zip(y_up_all, y_down_all, m.output_list)
                get!(raw_derivatives[o], k, Dict())
                raw_derivatives[o][k][-1] = (y_up - y_down) / (2*h)
            end
        elseif la == 0 && le == 0
            for y_up, y_down, o in zip(y_up_all, y_down_all, m.output_list)
                get!(raw_derivatives[o], k, Dict())
                raw_derivatives[o][k][0] = (y_up - y_down) / (2*h)
            end
        elseif la==0 && le==1
            for y_up, y_down, o in zip(y_up_all, y_down_all, m.output_list)
                get!(raw_derivatives[o], k, Dict())
                raw_derivatives[o][k][1] = (y_up - y_down) / (2*h)
            end
        else
            error("Lags or leads of more than 1, or both equal 1 are not supported")
        end
    end 

    J = Dict(o => Dict() for o in m.output_list)
    for o in m.output_list
        for k in keys(raw_derivatives[o])
            if T === nothing
                J[o][k] = from_simple_diagonals(raw_derivatives[o][k])
            else
                J[o][k] = from_simple_diagonals(raw_derivatives[o][k]).matrix(T)
            end
        end
    end
    return J
end

ex0= quote
    function firm(K,L,Z,α,δ)
        r = α*Z*(K(-1)/L)^(α-1) - δ
        w = (1-α)*Z*(K(-1)/L)^α
        Y = Z*K(-1)^α * L^(1-α)

        r,w,Y
    end
end

exx= quote
    r = α*Z*(K(-1)/L)^(α-1) - δ
    w = (1-α)*Z*(K(-1)/L)^α
    Y = Z*K(-1)^α * L^(1-α)
end

# Converts a dynamic expression to its static equivalent
to_static(e::Number, dynvars::Vector) = e
to_static(e::Symbol, dynvars::Vector) = e
function to_static(e::Expr, dynvars::Vector)
    @assert e.head == :call
    if e.args[1] in dynvars
        @assert length(e.args) == 2
        @assert isa(e.args[2], Integer)
        # at this step, we need to record non-zero time displacements i
        # with which k(i) appears in SimpleBlock
        e.args[1]
    else
        Expr(:call, e.args[1], map(x -> to_static(x, dynvars), e.args[2:end])...)
    end
end

# Replace an expression a=b by a-b
homogeneize(e::Number) = e
homogeneize(e::Symbol) = e
function homogeneize(e::Expr)
    if e.head == :(=)
        Expr(:call, :-, e.args...)
    else
        e
    end
end

@macroexpand @simple begin function firm(K,L,Z,α,δ)
    r = α*Z*(K(-1)/L)^(α-1) - δ
    w = (1-α)*Z*(K(-1)/L)^α
    Y = Z*K(-1)^α * L^(1-α)

    r,w,Y
end
end

# Fills a dictionnary which associates to each symbol its max lead and max lag
scan_expr(e::Number, endo::Vector{Symbol}, lead_lags::Dict{Symbol,Tuple{Int,Int}}) = nothing
scan_expr(e::Symbol, endo::Vector{Symbol}, lead_lags::Dict{Symbol,Tuple{Int,Int}}) = nothing
function scan_expr(e::Expr, endo::Vector{Symbol}, lag_lead::Dict{Symbol,Tuple{Int,Int}})
    if e.head == :call
        if !isempty(filter(x->x==e.args[1], endo))
            v = e.args[1]
            shift = e.args[2]
            @assert isa(shift, Integer)
            (la, le) = lag_lead[v]
            la = min(shift, la)
            le = max(shift, le)
            lag_lead[v] = (la, le)
        else
            map(x->scan_expr(x,endo,lag_lead), e.args[2:end])
        end
    elseif e.head == :(=)
        map(x->scan_expr(x,endo,lag_lead), e.args)
    else
        error("Unknown head")
    end
end
