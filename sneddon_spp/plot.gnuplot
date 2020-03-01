set size 1.0,1.0

set terminal postscript eps enhanced color dashed "Helvetica" 20

############################
# line styles              #
############################
# http://colorbrewer2.org/?type=qualitative&scheme=Set1&n=9
# red:
set style line 1  linetype 1 linecolor rgb "#e41a1c"  linewidth 3.000 pointtype 4 pointsize 2.0
# blue:
set style line 2  linetype 1 linecolor rgb "#377eb8"  linewidth 3.000 pointtype 6 pointsize 2.0
# green:
set style line 3  linetype 1 linecolor rgb "#4daf4a"  linewidth 3.000 pointtype 8 pointsize 2.0
# purple:
set style line 4  linetype 1 linecolor rgb "#984ea3"   linewidth 3.000 pointtype 9 pointsize 2.0
# orange:
set style line 5  linetype 5 dt 2 linecolor rgb "#ff7f00"   linewidth 3.000 pointtype 11 pointsize 2.0
# yellow:
set style line 6  linetype 5 dt 3 linecolor rgb "#ffff33"   linewidth 3.000 pointtype 5 pointsize 2.0
# brown
set style line 7  linetype 8 dt 4 linecolor rgb "#a65628"   linewidth 3.000 pointtype 8 pointsize 2.0
# pink
set style line 8  linetype 8 dt 5 linecolor rgb "#f781bf"   linewidth 3.000 pointtype 8 pointsize 2.0
# grey:
set style line 9  linetype 4 linecolor rgb "#999999"    linewidth 4.000 pointtype 1 pointsize 0.0
# black:
set style line 10  linetype 1 linecolor rgb "black"    linewidth 2.000 pointtype 1 pointsize 0.0



set xlabel "x"
set ylabel "COD"

set key top right
set key bottom center
#set term post eps enhanced color
#set terminal postscript eps 16

set output "cod_plot.eps"


max(x, y) = (x > y ? x : y)

# [-1.5:1.5][0:0.0007]
plot [-1.5:1.5] \
"cod-03b.txt" w lp title "h=1/8"  ls 1 lw 2,\
"cod-04b.txt" w lp title "h=1/16" ls 2 lw 2,\
"cod-05b.txt" w lp title "h=1/32" ls 3 lw 2,\
"cod-06b.txt" w lp title "h=1/64" ls 4 lw 2, \
1.92e-3*sqrt(max(0,1-x*x)) ls 10 title "exact"

!epstopdf cod_plot.eps
