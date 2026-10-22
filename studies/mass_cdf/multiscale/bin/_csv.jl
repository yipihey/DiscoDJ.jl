using DataFrames
function splitcsv(line)
    out = String[]; buf = IOBuffer(); q = false
    for c in line
        if c == '"'; q = !q
        elseif c == ',' && !q; push!(out, String(take!(buf)))
        else write(buf, c); end
    end
    push!(out, String(take!(buf))); return out
end
function parsecell(s)
    isempty(s) && return missing
    v = tryparse(Int, s); v !== nothing && return v
    v = tryparse(Float64, s); v !== nothing && return v
    s == "NaN" && return NaN
    return s
end
function readcsv(path)
    ls = readlines(path); hdr = splitcsv(ls[1])
    cols = [Any[] for _ in hdr]
    for l in ls[2:end]
        isempty(strip(l)) && continue
        f = splitcsv(l)
        for (i, x) in enumerate(f); push!(cols[i], parsecell(x)); end
    end
    return DataFrame([h => identity.(c) for (h, c) in zip(hdr, cols)])
end
