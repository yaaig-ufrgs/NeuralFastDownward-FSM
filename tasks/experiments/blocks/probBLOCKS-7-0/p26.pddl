(define (problem blocks-7-0)
(:domain blocks)
(:objects c f a b g d e )
(:init
	(clear b)
	(clear c)
	(clear f)
	(handempty)
	(on a e)
	(on b d)
	(on f g)
	(on g a)
	(ontable c)
	(ontable d)
	(ontable e)
)
(:goal (and (on a g) (on g d) (on d b) (on b c) (on c f) (on f e)))
)