(define (problem blocks-14-0)
(:domain blocks)
(:objects i d b l c k m h j n e f g a )
		(:init
		(clear a)
		(clear h)
		(clear m)
		(handempty)
		(on a g)
		(on b e)
		(on c l)
		(on d i)
		(on g d)
		(on h j)
		(on i n)
		(on j f)
		(on k c)
		(on l b)
		(on m k)
		(ontable e)
		(ontable f)
		(ontable n)
		)
(:goal (and (on e l) (on l f) (on f b) (on b j) (on j i) (on i n) (on n c)
            (on c k) (on k g) (on g d) (on d m) (on m a) (on a h)))
)
