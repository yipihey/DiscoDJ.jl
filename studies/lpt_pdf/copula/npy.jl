# Minimal .npy reader / writer (little-endian float64, C order) for the numpy <-> Julia hand-off.
function read_npy(path)
    open(path) do io
        magic = read(io, 6); @assert magic == UInt8[0x93, 'N', 'U', 'M', 'P', 'Y']
        major = read(io, UInt8); read(io, UInt8)
        hlen = major == 1 ? Int(read(io, UInt16)) : Int(read(io, UInt32))
        hdr = String(read(io, hlen))
        occursin("'<f8'", hdr) || error("expected little-endian float64: $hdr")
        occursin("'fortran_order': False", hdr) || error("expected C order")
        shp = Tuple(parse.(Int, split(match(r"\(([^)]*)\)", hdr)[1], ",", keepempty=false)))
        raw = Array{Float64}(undef, reverse(shp)...)
        read!(io, raw)
        return raw                                   # Julia dims = reversed numpy dims
    end
end

"""write_npy(path, A): A's Julia dims are the *reversed* numpy shape (A[k,j,i,c] ↔ numpy [c,i,j,k])."""
function write_npy(path, A::AbstractArray{<:Real})
    shp = reverse(size(A))
    hdr = "{'descr': '<f8', 'fortran_order': False, 'shape': (" * join(shp, ", ") * (length(shp) == 1 ? ",), }" : "), }")
    pad = 64 - (10 + length(hdr) + 1) % 64
    hdr = hdr * " "^pad * "\n"
    open(path * ".tmp", "w") do io
        write(io, UInt8[0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0]); write(io, UInt16(length(hdr))); write(io, hdr)
        write(io, Array{Float64}(A))
    end
    mv(path * ".tmp", path; force=true)
end
