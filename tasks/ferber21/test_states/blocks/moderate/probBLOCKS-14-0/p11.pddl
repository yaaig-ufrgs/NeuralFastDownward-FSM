(define (problem blocks-14-0)
(:domain blocks)
(:objects i d b l c k m h j n e f g a )
		(:init
		(clear c)
		(clear e)
		(clear h)
		(holding l)
		(on a g)
		(on b j)
		(on d i)
		(on e b)
		(on f a)
		(on g d)
		(on h m)
		(on i n)
		(on j k)
		(on k f)
		(ontable c)
		(ontable m)
		(ontable n)
		)
(:goal (and (on e l) (on l f) (on f b) (on b j) (on j i) (on i n) (on n c)
            (on c k) (on k g) (on g d) (on d m) (on m a) (on a h)))
)
