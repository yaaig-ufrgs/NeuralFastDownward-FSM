(define (problem blocks-14-0)
(:domain blocks)
(:objects i d b l c k m h j n e f g a )
		(:init
		(clear a)
		(clear g)
		(clear h)
		(clear i)
		(clear k)
		(handempty)
		(on a j)
		(on b e)
		(on c l)
		(on f d)
		(on h c)
		(on i m)
		(on j f)
		(on l b)
		(on m n)
		(ontable d)
		(ontable e)
		(ontable g)
		(ontable k)
		(ontable n)
		)
(:goal (and (on e l) (on l f) (on f b) (on b j) (on j i) (on i n) (on n c)
            (on c k) (on k g) (on g d) (on d m) (on m a) (on a h)))
)
