module PrettyBroadcastTree

import AbstractTrees


"""
This function may not be stable since it just uses the internal, private
Jula function that is intended to trim stack-traces. 
"""
function strip_excessive_types(str::String, n::Int)
    Base.type_depth_limit(str, n)
end

pp_typeof(x) = strip_excessive_types(string(typeof(x)), 120)


struct WrappedNode
    parent::Base.AbstractBroadcasted
    wrapped_args::Tuple
end

struct WrappedLeaf{T}
    value::T
end


function WrappedNode(bc::Base.AbstractBroadcasted)
    wrapped_args = map(
        arg -> isa(arg, Base.AbstractBroadcasted) ? WrappedNode(arg) : WrappedLeaf(arg),
        bc.args,
    )
    WrappedNode(bc, wrapped_args)
end


AbstractTrees.children(node::WrappedNode) = node.wrapped_args
AbstractTrees.children(::WrappedLeaf) = ()

# Printing of bullet lists
bullet_item(str::String) = "\n • $str"
print_bullet_item(io::IO, str::String) = print(io, bullet_item(str))

# Allow to fail getting a property via Optional
try_getproperty(obj, sym) = hasproperty(obj, sym) ? getproperty(obj, sym) : nothing


function AbstractTrees.printnode(io::IO, node::WrappedNode)
    print(io, pp_typeof(node.parent))

    # Try to get either a function or an operator
    op = something(
        try_getproperty(node.parent, :op),
        try_getproperty(node.parent, :f),
        Some(nothing),
    )
    op_type_str = op === nothing ? "No ':op' or ':f' property" : pp_typeof(op)
    print_bullet_item(io, "op: $(op_type_str)")

    # Axes should always be present but let's have a fallback
    axes = try_getproperty(node.parent, :axes)
    axes_type_str = axes === nothing ? "No ':axes' property" : pp_typeof(axes)
    print_bullet_item(io, "axes: $(axes_type_str)")
end

function AbstractTrees.printnode(io::IO, leaf::WrappedLeaf)
    print(io, pp_typeof(leaf.value))
end


"""
    pprint(bc::Base.Broadcasted)

Pretty-print the tree of a broadcasted expressions.
It trims the type parameters 
"""
pprint(bc::Base.AbstractBroadcasted) = (AbstractTrees.print_tree(WrappedNode(bc)); nothing)


export pprint


end
