(define (problem blocks-14-0)
(:domain blocks)
(:objects i d b l c k m h j n e f g a )
		(:init
		(clear f)
		(clear g)
		(clear h)
		(clear i)
		(clear m)
		(handempty)
		(on b e)
		(on c l)
		(on d a)
		(on g n)
		(on h j)
		(on i d)
		(on k c)
		(on l b)
		(on m k)
		(ontable a)
		(ontable e)
		(ontable f)
		(ontable j)
		(ontable n)
		)
(:goal (and (on e l) (on l f) (on f b) (on b j) (on j i) (on i n) (on n c)
            (on c k) (on k g) (on g d) (on d m) (on m a) (on a h)))
)
