(define (problem blocks-14-0)
(:domain blocks)
(:objects i d b l c k m h j n e f g a )
		(:init
		(clear a)
		(clear m)
		(holding f)
		(on a j)
		(on b e)
		(on c l)
		(on d g)
		(on h i)
		(on i d)
		(on j n)
		(on k c)
		(on l b)
		(on m k)
		(on n h)
		(ontable e)
		(ontable g)
		)
(:goal (and (on e l) (on l f) (on f b) (on b j) (on j i) (on i n) (on n c)
            (on c k) (on k g) (on g d) (on d m) (on m a) (on a h)))
)
