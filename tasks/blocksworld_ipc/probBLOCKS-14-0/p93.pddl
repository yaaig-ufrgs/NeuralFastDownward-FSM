(define (problem blocks-14-0)
(:domain blocks)
(:objects i d b l c k m h j n e f g a )
		(:init
		(clear a)
		(clear i)
		(handempty)
		(on a h)
		(on b e)
		(on c l)
		(on d j)
		(on f g)
		(on g d)
		(on h f)
		(on j n)
		(on k c)
		(on l b)
		(on m k)
		(on n m)
		(ontable e)
		(ontable i)
		)
(:goal (and (on e l) (on l f) (on f b) (on b j) (on j i) (on i n) (on n c)
            (on c k) (on k g) (on g d) (on d m) (on m a) (on a h)))
)
