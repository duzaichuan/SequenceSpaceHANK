using Optim
include("utils.jl")

function backward_iterate(Va_p, Pi_p, a_grid, e_grid, r, w, beta, eis)
    uc_nextgrid = (beta*Pi_p) * Va_p
    c_nextgrid = uc_nextgrid^(-eis)
    mesh_a = repeat(reshape(a_grid,(1,length(a_grid))), outer=[length(e_grid),1])
    mesh_e = repeat(reshape(e_grid, (length(e_grid),1)), outer=[1,length(a_grid)])
    coh = (1+r) * mesh_a + w * mesh_e
    a = utils.interpolate_y(c_nextgrid + mesh_a, coh, a_grid)

    c = coh -a
    Va = (1+r) * c^(-1/eis)
    return Va, a, c
end
