(define (problem blocks-14-0)
(:domain blocks)
(:objects i d b l c k m h j n e f g a )
		(:init
		(clear b)
		(clear h)
		(clear i)
		(clear n)
		(holding l)
		(on a g)
		(on b e)
		(on c m)
		(on f c)
		(on h f)
		(on i j)
		(on j a)
		(on m d)
		(on n k)
		(ontable d)
		(ontable e)
		(ontable g)
		(ontable k)
		)
(:goal (and (on e l) (on l f) (on f b) (on b j) (on j i) (on i n) (on n c)
            (on c k) (on k g) (on g d) (on d m) (on m a) (on a h)))
)
