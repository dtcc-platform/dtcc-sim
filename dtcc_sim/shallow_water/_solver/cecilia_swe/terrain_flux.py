"""Experimental hydrostatic DG flux with building reflections and semi-dry balance."""
import ufl


def build_rhs_form(domain,V,U,bed,gravity,floor,semidry,wall_facets=None,*,equilibrium_depth=None):
    W=ufl.TestFunction(V)
    normal=ufl.FacetNormal(domain)
    n=ufl.as_vector([normal[0],normal[1]])
    grad=ufl.as_tensor([[W[k].dx(j) for j in range(2)] for k in range(3)])
    def state(q,b,top):
        h=ufl.max_value(q[0],0.)
        d=ufl.max_value(h+b-top,0.)
        v=ufl.as_vector([q[1],q[2]])/ufl.max_value(h,floor)
        effective=ufl.as_vector([d,d*v[0],d*v[1]])
        p=.5*gravity*d*d
        flux=ufl.as_tensor([[d*v[0],d*v[1]],
            [d*v[0]*v[0]+p,d*v[0]*v[1]], [d*v[0]*v[1],d*v[1]*v[1]+p]])
        return effective,flux,v,h,d
    def interface(a,b,za,zb,n):
        top=ufl.max_value(za,zb)
        qa,fa,va,ha,da=state(a,za,top)
        qb,fb,vb,hb,db=state(b,zb,top)
        speed=ufl.max_value(abs(ufl.dot(va,n))+ufl.sqrt(gravity*da),abs(ufl.dot(vb,n))+ufl.sqrt(gravity*db))
        common=.5*ufl.dot(fa+fb,n)-.5*speed*(qb-qa)
        direction=ufl.as_vector([0.,n[0],n[1]])
        return common+.5*gravity*(ha*ha-da*da)*direction,common+.5*gravity*(hb*hb-db*db)*direction
    def reflection(q,n):
        momentum=ufl.as_vector([q[1],q[2]])
        reflected=momentum-2*ufl.dot(momentum,n)*n
        return ufl.as_vector([q[0],reflected[0],reflected[1]])
    def pressure_trace(q,n):
        return .5*gravity*q[0]**2*ufl.as_vector([0.,n[0],n[1]])
    _,physical,_,_,_=state(U,bed,bed)
    equilibrium_state=U if equilibrium_depth is None else ufl.as_vector([equilibrium_depth,0.,0.])
    pressure=.5*gravity*equilibrium_state[0]**2
    pressure_tensor=ufl.as_tensor([[0.,0.],[pressure,0.],[0.,pressure]])
    source=ufl.as_vector([0.,-gravity*U[0]*bed.dx(0),-gravity*U[0]*bed.dx(1)])
    equilibrium_source=ufl.as_vector([0.,-gravity*equilibrium_state[0]*bed.dx(0),-gravity*equilibrium_state[0]*bed.dx(1)])
    balanced_source=(1-semidry)*source if equilibrium_depth is None else source-semidry*equilibrium_source
    metadata={'quadrature_degree':6}
    dx=ufl.Measure('dx',domain=domain,metadata=metadata)
    ds=ufl.Measure('ds',domain=domain,metadata=metadata)
    dS=ufl.Measure('dS',domain=domain,metadata=metadata,subdomain_data=wall_facets)
    exterior,_=interface(U,reflection(U,n),bed,bed,n)
    rhs=ufl.inner(physical-semidry*pressure_tensor,grad)*dx+ufl.inner(balanced_source,W)*dx
    rhs-=ufl.inner(exterior-semidry*pressure_trace(equilibrium_state,n),W)*ds
    left,right=interface(U('-'),U('+'),bed('-'),bed('+'),n('-'))
    interior=(-ufl.inner(left-semidry('-')*pressure_trace(equilibrium_state('-'),n('-')),W('-'))
              +ufl.inner(right-semidry('+')*pressure_trace(equilibrium_state('+'),n('-')),W('+')))
    rhs+=interior*(dS if wall_facets is None else dS(0))
    if wall_facets is not None:
        for side in ('-','+'):
            flux,_=interface(U(side),reflection(U(side),n(side)),bed(side),bed(side),n(side))
            rhs-=ufl.inner(flux-semidry(side)*pressure_trace(equilibrium_state(side),n(side)),W(side))*dS(1)
    return rhs
