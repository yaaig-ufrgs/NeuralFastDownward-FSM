(define (problem blocks-14-0)
(:domain blocks)
(:objects i d b l c k m h j n e f g a )
		(:init
		(clear c)
		(clear d)
		(clear h)
		(clear j)
		(clear m)
		(holding k)
		(on a n)
		(on b e)
		(on d i)
		(on f a)
		(on g f)
		(on j g)
		(on l b)
		(on m l)
		(ontable c)
		(ontable e)
		(ontable h)
		(ontable i)
		(ontable n)
		)
(:goal (and (on e l) (on l f) (on f b) (on b j) (on j i) (on i n) (on n c)
            (on c k) (on k g) (on g d) (on d m) (on m a) (on a h)))
)
