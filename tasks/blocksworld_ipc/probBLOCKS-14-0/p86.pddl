(define (problem blocks-14-0)
(:domain blocks)
(:objects i d b l c k m h j n e f g a )
		(:init
		(clear c)
		(clear d)
		(clear g)
		(clear i)
		(clear m)
		(clear n)
		(holding j)
		(on a f)
		(on b e)
		(on c b)
		(on d k)
		(on h a)
		(on i l)
		(on m h)
		(ontable e)
		(ontable f)
		(ontable g)
		(ontable k)
		(ontable l)
		(ontable n)
		)
(:goal (and (on e l) (on l f) (on f b) (on b j) (on j i) (on i n) (on n c)
            (on c k) (on k g) (on g d) (on d m) (on m a) (on a h)))
)
