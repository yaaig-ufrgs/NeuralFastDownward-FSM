(define (problem blocks-14-0)
(:domain blocks)
(:objects i d b l c k m h j n e f g a )
		(:init
		(clear d)
		(clear f)
		(clear i)
		(clear k)
		(clear n)
		(handempty)
		(on b e)
		(on c l)
		(on d c)
		(on f j)
		(on h g)
		(on i a)
		(on j h)
		(on l b)
		(on n m)
		(ontable a)
		(ontable e)
		(ontable g)
		(ontable k)
		(ontable m)
		)
(:goal (and (on e l) (on l f) (on f b) (on b j) (on j i) (on i n) (on n c)
            (on c k) (on k g) (on g d) (on d m) (on m a) (on a h)))
)