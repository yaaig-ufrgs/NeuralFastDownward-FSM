(define (problem blocks-14-0)
(:domain blocks)
(:objects i d b l c k m h j n e f g a )
		(:init
		(clear e)
		(clear h)
		(clear j)
		(clear k)
		(holding b)
		(on a g)
		(on c l)
		(on d i)
		(on f d)
		(on g f)
		(on h m)
		(on i n)
		(on l a)
		(on m c)
		(ontable e)
		(ontable j)
		(ontable k)
		(ontable n)
		)
(:goal (and (on e l) (on l f) (on f b) (on b j) (on j i) (on i n) (on n c)
            (on c k) (on k g) (on g d) (on d m) (on m a) (on a h)))
)
