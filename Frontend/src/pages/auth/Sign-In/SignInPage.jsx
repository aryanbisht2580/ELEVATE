import { SignIn } from '@clerk/clerk-react'
import { useUser } from '@clerk/clerk-react';
import React, { useEffect } from 'react';
import { useDispatch } from 'react-redux';
import { addUserData } from '@/features/user/userFeatures';
import { useNavigate } from 'react-router-dom';

function SignInPage() {
  const { user, isSignedIn } = useUser();
  const dispatch = useDispatch();
  const navigate = useNavigate();

  useEffect(() => {
    if (isSignedIn && user) {
      // Convert Clerk user data to match your app's format
      const userData = {
        id: user.id,
        email: user.primaryEmailAddress?.emailAddress,
        fullName: user.fullName || `${user.firstName || ''} ${user.lastName || ''}`.trim(),
        // Add other fields as needed
      };
      dispatch(addUserData(userData));
    }
  }, [isSignedIn, user, dispatch]);

  return (
    <div className=' flex justify-center items-center py-20'>
      <SignIn 
        afterSignInUrl="/"
        routing="virtual"
      />
    </div>
  )
}

export default SignInPage