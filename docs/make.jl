using DNNS
import Pkg

Pkg.add("Documenter")
using Documenter

makedocs(
	sitename = "DNNS",
	format = Documenter.HTML(),
	modules = [DNNS]
	)

	# Documenter can also automatically deploy documentation to gh-pages.
	# See "Hosting Documentation" and deploydocs() in the Documenter manual
	# for more information.
	deploydocs(
		repo = "github.com/scottrsm/DNNS.jl.git"
	)
