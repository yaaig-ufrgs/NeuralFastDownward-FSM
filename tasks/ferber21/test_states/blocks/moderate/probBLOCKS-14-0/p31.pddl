(define (problem blocks-14-0)
(:domain blocks)
(:objects i d b l c k m h j n e f g a )
		(:init
		(clear a)
		(clear j)
		(holding h)
		(on a g)
		(on b e)
		(on c l)
		(on d i)
		(on f k)
		(on g d)
		(on i n)
		(on j m)
		(on k c)
		(on l b)
		(on m f)
		(ontable e)
		(ontable n)
		)
(:goal (and (on e l) (on l f) (on f b) (on b j) (on j i) (on i n) (on n c)
            (on c k) (on k g) (on g d) (on d m) (on m a) (on a h)))
)
