(define (problem blocks-14-0)
(:domain blocks)
(:objects i d b l c k m h j n e f g a )
		(:init
		(clear c)
		(clear d)
		(clear g)
		(clear h)
		(clear k)
		(holding a)
		(on b j)
		(on c e)
		(on e b)
		(on f i)
		(on g f)
		(on i n)
		(on j l)
		(on n m)
		(ontable d)
		(ontable h)
		(ontable k)
		(ontable l)
		(ontable m)
		)
(:goal (and (on e l) (on l f) (on f b) (on b j) (on j i) (on i n) (on n c)
            (on c k) (on k g) (on g d) (on d m) (on m a) (on a h)))
)
