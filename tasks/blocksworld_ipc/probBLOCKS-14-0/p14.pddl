(define (problem blocks-14-0)
(:domain blocks)
(:objects i d b l c k m h j n e f g a )
		(:init
		(clear c)
		(clear e)
		(clear k)
		(clear n)
		(holding m)
		(on b d)
		(on c l)
		(on d a)
		(on e b)
		(on g i)
		(on h j)
		(on j f)
		(on k g)
		(on l h)
		(ontable a)
		(ontable f)
		(ontable i)
		(ontable n)
		)
(:goal (and (on e l) (on l f) (on f b) (on b j) (on j i) (on i n) (on n c)
            (on c k) (on k g) (on g d) (on d m) (on m a) (on a h)))
)
